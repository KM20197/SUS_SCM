"""
Google Colab (duas células):

1) Instale dependências:
   !pip install -q deap statsmodels scikit-optimize tensorflow pandas numpy scipy scikit-learn python-docx tqdm

2) Execute com progresso, sem buffer de stdout (recomendado):
   !python -u Code4_colab_fixed_v12_progress_checkpoint_sigint.py --progress --log_every 50 --checkpoint_every 2000

Notas:
- Este script não altera a lógica do experimento. Ele apenas adiciona telemetria e checkpointing de CSV.
- O arquivo parcial "raw_replications_partial.csv" permite acompanhar resultados e recuperar execuções interrompidas.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
# Suppress TensorFlow C++ logs (verbosity only; no effect on results)
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import random
import sys
import time
import signal
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from time import perf_counter
from datetime import timedelta

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

# ML / DL
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras import Input, Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# GA
from deap import base, creator, tools


# ----------------------------------------------------------------------------
# Optional exception telemetry (verbosity only; no effect on results)
# ----------------------------------------------------------------------------

LOG_EXCEPTIONS: bool = False  # set from CLI (--progress) in main()
_EXC_COUNTS: Dict[str, int] = {}
_EXC_MAX_PRINTS: int = 3


def _log_exception(tag: str, exc: BaseException) -> None:
    """Prints a throttled warning when --progress is enabled.

    This function does not alter any experiment parameters or computations.
    """
    if not LOG_EXCEPTIONS:
        return
    n = _EXC_COUNTS.get(tag, 0)
    _EXC_COUNTS[tag] = n + 1
    if n < _EXC_MAX_PRINTS:
        print(f"[warn] {tag}: {type(exc).__name__}: {exc}")
        sys.stdout.flush()


# ----------------------------------------------------------------------------
# Reproducibility fingerprint (telemetry only)
# ----------------------------------------------------------------------------

def _sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """SHA256 of a file (streaming). Telemetry only; does not affect results."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def write_run_fingerprint(
    out_dir: Path,
    raw_path: Path,
    output_csv_paths: List[Path],
    manifest_path: Optional[Path] = None,
    filename: str = "run_fingerprint.json",
) -> Path:
    """Write a compact fingerprint JSON for quick run-to-run comparisons."""
    fp = {
        "raw_path": str(raw_path),
        "raw_sha256": _sha256_file(raw_path) if raw_path.exists() else None,
        "outputs": {
            p.name: _sha256_file(p) for p in output_csv_paths if isinstance(p, Path) and p.exists()
        },
        "manifest": {
            "name": manifest_path.name if manifest_path else None,
            "sha256": _sha256_file(manifest_path) if (manifest_path and manifest_path.exists()) else None,
        },
        "environment": {
            "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        },
    }
    fp_path = out_dir / filename
    fp_path.write_text(json.dumps(fp, indent=2, sort_keys=True), encoding="utf-8")
    return fp_path

# -----------------------------------------------------------------------------
# Repro
# -----------------------------------------------------------------------------

def set_global_seeds(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)


# ----------------------------------------------------------------------------
# Deterministic 32-bit seed from arbitrary parts (does not depend on Python's hash())
# ----------------------------------------------------------------------------
def seed32(*parts: object) -> int:
    s = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(s, digest_size=8).digest(), "little") & 0xFFFFFFFF

# -----------------------------------------------------------------------------
# Utility: Holm correction (within a family)
# -----------------------------------------------------------------------------

def holm_adjust(pvals: Sequence[float]) -> List[float]:
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m, dtype=float)
    running = 0.0
    for k, i in enumerate(order):
        running = max(running, (m - k) * float(pvals[i]))
        adj[i] = min(1.0, running)
    return adj.tolist()

# -----------------------------------------------------------------------------
# Forecasting methods
# -----------------------------------------------------------------------------

class Forecaster:
    def fit(self, y: np.ndarray) -> None:
        raise NotImplementedError
    def predict_one(self, hist: np.ndarray) -> float:
        raise NotImplementedError
    @property
    def name(self) -> str:
        return self.__class__.__name__.lower()

class SESForecaster(Forecaster):
    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha
        self.level = None

    def fit(self, y: np.ndarray) -> None:
        y = np.asarray(y, dtype=float)
        self.level = float(np.mean(y)) if y.size else 0.0
        if y.size:
            lvl = float(y[0])
            for x in y[1:]:
                lvl = lvl + self.alpha * (float(x) - lvl)
            self.level = lvl

    def predict_one(self, hist: np.ndarray) -> float:
        if self.level is None:
            self.fit(hist)
        return max(0.0, float(self.level))

    @property
    def name(self) -> str:
        return "ses"

class HoltForecaster(Forecaster):
    def __init__(self, alpha: float = 0.3, beta: float = 0.3):
        self.alpha = alpha
        self.beta = beta
        self.level = None
        self.trend = None

    def fit(self, y: np.ndarray) -> None:
        y = np.asarray(y, dtype=float)
        if y.size == 0:
            self.level, self.trend = 0.0, 0.0
            return
        lvl = float(y[0])
        tr = float(y[1] - y[0]) if y.size >= 2 else 0.0
        for x in y[1:]:
            prev_lvl = lvl
            lvl = lvl + tr + self.alpha * (float(x) - (lvl + tr))
            tr = tr + self.beta * (lvl - prev_lvl - tr)
        self.level, self.trend = lvl, tr

    def predict_one(self, hist: np.ndarray) -> float:
        if self.level is None or self.trend is None:
            self.fit(hist)
        return max(0.0, float(self.level + self.trend))

    @property
    def name(self) -> str:
        return "holt"

class CrostonForecaster(Forecaster):
    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha

    def fit(self, y: np.ndarray) -> None:
        # Stateless; we compute from history each time (short series).
        return

    def predict_one(self, hist: np.ndarray) -> float:
        y = np.asarray(hist, dtype=float)
        if y.size == 0:
            return 0.0
        nz = np.where(y > 0)[0]
        if nz.size == 0:
            return 0.0

        z = float(y[nz[0]])
        p = 1.0
        last_nz = nz[0]
        for t in range(1, y.size):
            if y[t] > 0:
                z = z + self.alpha * (float(y[t]) - z)
                tau = float(t - last_nz)
                p = p + self.alpha * (tau - p)
                last_nz = t
        f = z / max(p, 1e-9)
        return max(0.0, float(f))

    @property
    def name(self) -> str:
        return "croston"

class LSTMRFEnsemble(Forecaster):
    """
    Code3-style ensemble. For short histories, it falls back to mean demand.
    """
    def __init__(self, seq_len: int = 14, lstm_units: int = 64, dropout_rate: float = 0.2):
        self.seq_len = seq_len
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.lstm = None
        self.rf = None
        self.fallback = 0.0
        self.is_fitted = False

    def _create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for i in range(self.seq_len, len(data)):
            X.append(data[i - self.seq_len : i])
            y.append(data[i])
        return np.asarray(X, dtype=float), np.asarray(y, dtype=float)

    def fit(self, y: np.ndarray) -> None:
        data = np.asarray(y, dtype=float)
        self.fallback = float(np.mean(data)) if data.size else 0.0

        if data.size < self.seq_len * 2:
            self.is_fitted = False
            return

        X, y2 = self._create_sequences(data)
        idx = int(0.8 * len(X))
        Xtr, Xv = X[:idx], X[idx:]
        ytr, yv = y2[:idx], y2[idx:]
        if Xv.size == 0:
            Xv, yv = Xtr[-1:], ytr[-1:]

        Xtr_s = self.scaler_X.fit_transform(Xtr.reshape(-1, self.seq_len)).reshape(Xtr.shape)
        Xv_s = self.scaler_X.transform(Xv.reshape(-1, self.seq_len)).reshape(Xv.shape)
        ytr_s = self.scaler_y.fit_transform(ytr.reshape(-1, 1)).flatten()
        yv_s = self.scaler_y.transform(yv.reshape(-1, 1)).flatten()

        self.lstm = Sequential([
            Input((self.seq_len, 1)),
            LSTM(self.lstm_units, activation="tanh", return_sequences=True),
            Dropout(self.dropout_rate),
            LSTM(self.lstm_units // 2, activation="tanh"),
            Dropout(self.dropout_rate),
            Dense(16, activation="relu"),
            Dense(1),
        ])
        self.lstm.compile(optimizer="adam", loss="mse")
        self.lstm.fit(
            Xtr_s.reshape(-1, self.seq_len, 1), ytr_s,
            validation_data=(Xv_s.reshape(-1, self.seq_len, 1), yv_s),
            epochs=40, batch_size=min(32, len(Xtr)),
            verbose=0,
            callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
        )

        self.rf = RandomForestRegressor(
            n_estimators=200, max_depth=15, min_samples_split=5, random_state=42
        )
        self.rf.fit(Xtr, ytr)
        self.is_fitted = True

    def predict_one(self, hist: np.ndarray) -> float:
        hist = np.asarray(hist, dtype=float)
        if (not self.is_fitted) or hist.size < self.seq_len:
            return max(0.0, float(np.mean(hist)) if hist.size else self.fallback)

        seq = hist[-self.seq_len :].astype(float)
        preds = []

        try:
            x_s = self.scaler_X.transform(seq.reshape(1, -1)).reshape(1, self.seq_len, 1)
            p1 = self.scaler_y.inverse_transform(self.lstm.predict(x_s, verbose=0))[0][0]
            preds.append(max(0.0, float(p1)))
        except Exception as e:
            _log_exception("lstm_predict_one", e)

        try:
            p2 = float(self.rf.predict(seq.reshape(1, -1))[0])
            preds.append(max(0.0, p2))
        except Exception as e:
            _log_exception("rf_predict_one", e)

        return float(np.mean(preds)) if preds else max(0.0, self.fallback)

    @property
    def name(self) -> str:
        return "lstm_rf"

def make_forecaster(name: str, alpha: float = 0.3) -> Forecaster:
    name = name.lower().strip()
    if name in {"ses"}:
        return SESForecaster(alpha=alpha)
    if name in {"holt"}:
        return HoltForecaster(alpha=alpha, beta=alpha)
    if name in {"croston"}:
        return CrostonForecaster(alpha=alpha)
    if name in {"lstm_rf", "lstm+rf", "lstmrf"}:
        return LSTMRFEnsemble()
    raise ValueError(f"Unknown forecaster: {name}")

# -----------------------------------------------------------------------------
# RL agent (Double Q-learning) with reward shaping
# -----------------------------------------------------------------------------

@dataclass
class RLConfig:
    n_states: int = 100
    n_actions: int = 7
    alpha: float = 0.1
    gamma: float = 0.95
    epsilon: float = 0.2
    epsilon_min: float = 0.02
    epsilon_decay: float = 0.995

class DoubleQLearningAgent:
    def __init__(self, cfg: RLConfig):
        self.cfg = cfg
        self.q1 = np.zeros((cfg.n_states, cfg.n_actions), dtype=float)
        self.q2 = np.zeros((cfg.n_states, cfg.n_actions), dtype=float)
        self.state_bins = None
        self.order_scale = 1.0
        self.epsilon = cfg.epsilon

    def _grid_size(self) -> int:
        return int(round(math.sqrt(self.cfg.n_states)))

    def init_bins(self, demand: np.ndarray, forecast: np.ndarray) -> None:
        # 10 bins each -> grid 10x10 -> 100 states (default)
        self.state_bins = {
            "inventory": np.percentile(demand * 3.0, np.linspace(0, 100, 11)),
            "forecast": np.percentile(forecast, np.linspace(0, 100, 11)),
        }
        self.order_scale = max(1.0, float(np.percentile(demand, 90)) / (self.cfg.n_actions - 1))

    def discretize(self, inventory: float, forecast: float) -> int:
        if self.state_bins is None:
            return 0
        g = self._grid_size()
        inv_bin = np.clip(np.digitize(inventory, self.state_bins["inventory"]) - 1, 0, g - 1)
        fore_bin = np.clip(np.digitize(forecast, self.state_bins["forecast"]) - 1, 0, g - 1)
        s = int(inv_bin * g + fore_bin)
        return min(max(s, 0), self.cfg.n_states - 1)

    def choose_action(self, state: int, eval_mode: bool = False) -> int:
        s = int(np.clip(state, 0, self.cfg.n_states - 1))
        if eval_mode or random.random() >= self.epsilon:
            return int(np.argmax((self.q1[s] + self.q2[s]) / 2.0))
        return random.randint(0, self.cfg.n_actions - 1)

    def update(self, state: int, action: int, reward: float, next_state: int) -> None:
        s = int(np.clip(state, 0, self.cfg.n_states - 1))
        a = int(np.clip(action, 0, self.cfg.n_actions - 1))
        ns = int(np.clip(next_state, 0, self.cfg.n_states - 1))

        if random.random() < 0.5:
            best = int(np.argmax(self.q1[ns]))
            target = reward + self.cfg.gamma * self.q2[ns, best]
            self.q1[s, a] += self.cfg.alpha * (target - self.q1[s, a])
        else:
            best = int(np.argmax(self.q2[ns]))
            target = reward + self.cfg.gamma * self.q1[ns, best]
            self.q2[s, a] += self.cfg.alpha * (target - self.q2[s, a])

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.cfg.epsilon_min, self.epsilon * self.cfg.epsilon_decay)

# -----------------------------------------------------------------------------
# Routing cost proxy with GA ablation
# -----------------------------------------------------------------------------

def _route_length(order: Sequence[int], coords: np.ndarray) -> float:
    # closed tour
    dist = 0.0
    for i in range(len(order) - 1):
        a, b = order[i], order[i+1]
        dist += float(np.linalg.norm(coords[a] - coords[b]))
    dist += float(np.linalg.norm(coords[order[-1]] - coords[order[0]]))
    return dist

def routing_cost_proxy(
    rng: np.random.Generator,
    order_qty: float,
    ga_on: bool,
    route_fixed: float,
    route_per_unit: float,
    capacity: float = 200.0,
    max_stops: int = 12,
    ga_timeout_s: float = 1.5,
) -> float:
    """
    Converts ordering action into a routing cost proxy.
    - Base proxy: route_fixed + route_per_unit * qty
    - GA on/off: GA computes an improvement factor based on a small TSP instance.
    """
    if order_qty <= 0:
        return 0.0

    base_cost = route_fixed + route_per_unit * float(order_qty)

    # Build a small TSP instance size ~ number of truck stops
    n_stops = int(np.clip(math.ceil(order_qty / capacity), 2, max_stops))
    coords = rng.normal(0.0, 1.0, size=(n_stops, 2)).astype(float)

    # Naive route: identity order
    naive = list(range(n_stops))
    naive_len = _route_length(naive, coords)

    if not ga_on:
        # map length ratio to cost multiplier (>=1.0)
        return float(base_cost) * float(naive_len / max(naive_len, 1e-9))

    # GA solve (time-bounded)
    if not hasattr(creator, "FitnessMinRouting"):
        creator.create("FitnessMinRouting", base.Fitness, weights=(-1.0,))
    if not hasattr(creator, "IndividualRouting"):
        creator.create("IndividualRouting", list, fitness=creator.FitnessMinRouting)

    tb = base.Toolbox()
    tb.register("indices", random.sample, range(n_stops), n_stops)
    tb.register("individual", tools.initIterate, creator.IndividualRouting, tb.indices)
    tb.register("population", tools.initRepeat, list, tb.individual)

    def eval_ind(ind):
        return (_route_length(ind, coords),)

    tb.register("evaluate", eval_ind)
    tb.register("mate", tools.cxOrdered)
    tb.register("mutate", tools.mutShuffleIndexes, indpb=0.2)
    tb.register("select", tools.selTournament, tournsize=3)

    pop = tb.population(n=40)
    hof = tools.HallOfFame(1)

    start = time.time()
    # short run
    gen = 0
    while gen < 60 and (time.time() - start) < ga_timeout_s:
        pop = tb.select(pop, len(pop))
        for i1, i2 in zip(pop[::2], pop[1::2]):
            if random.random() < 0.8:
                tb.mate(i1, i2)
                del i1.fitness.values
                del i2.fitness.values
        for ind in pop:
            if random.random() < 0.2:
                tb.mutate(ind)
                del ind.fitness.values
        invalid = [ind for ind in pop if not ind.fitness.valid]
        for ind in invalid:
            ind.fitness.values = tb.evaluate(ind)
        hof.update(pop)
        gen += 1

    best_len = float(hof[0].fitness.values[0]) if len(hof) else float(naive_len)
    # improvement factor <= 1.0
    mult = best_len / max(naive_len, 1e-9)
    return float(base_cost) * float(mult)

# -----------------------------------------------------------------------------
# Inventory environment with pipeline lead times
# -----------------------------------------------------------------------------

@dataclass
class CostParams:
    stockout_penalty: float = 25.0
    holding_rate: float = 0.003
    ordering_rate: float = 1.5
    service_bonus: float = 8.0
    safety_stock_factor: float = 2.0

@dataclass
class RouteParams:
    route_fixed: float = 120.0
    route_per_unit: float = 0.35

def lead_time_days(scenario: str, rng: np.random.Generator) -> int:
    scenario = scenario.upper()
    if scenario == "SUPPLY_DISRUPTION":
        return int(rng.integers(56, 71))  # 8–10 weeks
    if scenario == "SEASONAL_SURGE":
        return int(rng.integers(7, 15))
    if scenario == "HIGH_VOLATILITY":
        return int(rng.integers(5, 11))
    return int(rng.integers(4, 8))

def simulate_policy_episode(
    scenario: str,
    demand: np.ndarray,
    unit_costs: np.ndarray,
    policy: str,
    forecaster: Forecaster,
    agent: Optional[DoubleQLearningAgent],
    cost_params: CostParams,
    route_params: RouteParams,
    ga_on: bool,
    reward_mode: str,
    routing_weight: float,
    warmup_days: int,
    eval_days: int,
    carryover: str,
    rng: np.random.Generator,
) -> Dict[str, float]:
    """
    Single episode for one policy with pipeline inventory.
    carryover:
      - "truncate": ignore holding cost for pipeline arrivals after eval window
      - "carryover": extend with zero demand until pipeline empties; accrue holding on arrivals
    reward_mode:
      - "inventory_only"
      - "inventory_plus_routing"
    """
    policy = policy.upper()
    scenario = scenario.upper()
    carryover = carryover.lower().strip()
    reward_mode = reward_mode.lower().strip()

    total_days = warmup_days + eval_days
    assert demand.size >= total_days, "Demand series shorter than warmup+eval."

    # pipeline: list of (arrival_day, qty)
    pipeline: List[Tuple[int, float]] = []

    inv = float(np.mean(demand[:warmup_days]) * cost_params.safety_stock_factor) if warmup_days > 0 else float(np.mean(demand) * cost_params.safety_stock_factor)

    holding_cost = 0.0
    stockout_cost = 0.0
    ordering_cost = 0.0
    routing_cost = 0.0
    demand_total = 0.0
    fulfilled_total = 0.0
    stockout_days = 0

    # for policies needing targets
    mean_d = float(np.mean(demand[:warmup_days])) if warmup_days > 0 else float(np.mean(demand))
    std_d = float(np.std(demand[:warmup_days])) if warmup_days > 0 else float(np.std(demand))

    # forecast fitting: use warm-up history
    forecaster.fit(demand[:warmup_days] if warmup_days > 0 else demand[:max(14, min(60, total_days))])

    # base-stock target parameters
    z = 1.96

    def compute_order_qty(t: int, d_t: float, uc: float, hist: np.ndarray) -> float:
        nonlocal inv, mean_d, std_d
        f = float(forecaster.predict_one(hist))
        if policy == "STATIC":
            # fixed replenishment to mean level
            target = mean_d
            return float(target) if inv < mean_d else 0.0

        if policy == "DYNAMIC":
            recent = hist[max(0, len(hist)-7):]
            f2 = float(np.mean(recent)) if recent.size else f
            ss = 2.0 * float(np.std(recent)) if recent.size > 1 else 0.3 * mean_d
            target_level = f2 + ss
            return float(max(0.0, target_level - inv))

        if policy == "DYN_FC":
            # forecast-informed conventional baseline (parity): order-up-to based on forecast mean and forecast volatility
            recent = hist[max(0, len(hist)-14):]
            mu = float(f)
            sigma = float(np.std(recent)) if recent.size > 1 else float(std_d)
            lead = float(lead_time_days(scenario, rng))
            order_up_to = mu * lead + z * sigma * math.sqrt(max(lead, 1.0))
            return float(max(0.0, order_up_to - inv))

        if policy == "BASESTOCK":
            lead = float(lead_time_days(scenario, rng))
            mu = float(f)
            sigma = float(np.std(hist[-30:])) if hist.size > 1 else float(std_d)
            base = mu * lead + z * sigma * math.sqrt(max(lead, 1.0))
            return float(max(0.0, base - inv))

        if policy == "AI":
            if agent is None:
                return 0.0
            s = agent.discretize(inv, f)
            a = agent.choose_action(s, eval_mode=True)
            return float(a * agent.order_scale)

        raise ValueError(f"Unknown policy: {policy}")

    # simulation over warmup+eval
    for t in range(total_days):
        # arrivals
        if pipeline:
            arrived = [q for (ad, q) in pipeline if ad == t]
            if arrived:
                inv += float(np.sum(arrived))
            pipeline = [(ad, q) for (ad, q) in pipeline if ad != t]

        hist = demand[:t]  # realized history up to t-1
        d_t = float(demand[t])
        uc = float(unit_costs[t])

        # decision
        order_qty = compute_order_qty(t, d_t, uc, hist)
        if order_qty > 0:
            lt = int(lead_time_days(scenario, rng))
            pipeline.append((t + lt, float(order_qty)))

            ordering_cost += float(order_qty) * cost_params.ordering_rate

            # routing proxy (evaluated at order time)
            rc = routing_cost_proxy(
                rng=rng, order_qty=order_qty, ga_on=ga_on,
                route_fixed=route_params.route_fixed, route_per_unit=route_params.route_per_unit
            )
            routing_cost += float(rc)

        # demand realization (lost sales, no backorder)
        available = inv
        fulfilled = min(available, d_t)
        lost = max(0.0, d_t - fulfilled)

        inv = max(0.0, available - fulfilled)

        demand_total += d_t
        fulfilled_total += fulfilled
        if lost > 0:
            stockout_days += 1

        # costs for the day
        # stockout penalty scales with unit cost
        stockout_cost += lost * uc * cost_params.stockout_penalty + (lost ** 2) * 0.05
        holding_cost += inv * uc * cost_params.holding_rate

        # RL reward shaping (training only; during evaluation we compute costs regardless)
        # We do not train inside this function.

    # carryover: add holding cost for pipeline arrivals after end of eval window
    if carryover == "carryover" and pipeline:
        # Extend until last arrival
        last_arrival = max(ad for (ad, _q) in pipeline)
        # Guard: unit_costs should be non-empty for any positive horizon; keep safe anyway.
        uc_last = float(unit_costs[-1]) if len(unit_costs) else 0.0
        # accrue until last arrival (inclusive), with zero demand
        for t in range(total_days, last_arrival + 1):
            arrived = [q for (ad, q) in pipeline if ad == t]
            if arrived:
                inv += float(np.sum(arrived))
            pipeline = [(ad, q) for (ad, q) in pipeline if ad != t]
            # holding cost for carried inventory (use last unit cost as proxy)
            uc = uc_last
            holding_cost += inv * uc * cost_params.holding_rate

    total_cost = holding_cost + stockout_cost + ordering_cost + routing_cost
    fill_rate = fulfilled_total / max(1e-9, demand_total)
    stockout_rate = stockout_days / max(1, total_days)

    return {
        "total_cost": float(total_cost),
        "holding_cost": float(holding_cost),
        "stockout_cost": float(stockout_cost),
        "ordering_cost": float(ordering_cost),
        "routing_cost": float(routing_cost),
        "fill_rate": float(fill_rate),
        "stockout_rate": float(stockout_rate),
    }

# -----------------------------------------------------------------------------
# Demand generation (synthetic; optionally parameterized from data)
# -----------------------------------------------------------------------------

@dataclass
class DemandParams:
    mean: float
    sd: float

def estimate_params_from_data(df: pd.DataFrame, qty_col: str) -> DemandParams:
    q = pd.to_numeric(df[qty_col], errors="coerce").dropna().astype(float).values
    if q.size == 0:
        return DemandParams(mean=150.0, sd=63.0)
    return DemandParams(mean=float(np.mean(q)), sd=float(np.std(q)))

def generate_demand(scenario: str, n_rep: int, horizon: int, params: DemandParams, rng: np.random.Generator) -> np.ndarray:
    mu, sd = params.mean, params.sd
    scenario = scenario.upper()
    demands = np.zeros((n_rep, horizon), dtype=float)
    for r in range(n_rep):
        if scenario == "STABLE":
            d = rng.normal(mu, sd, horizon)
        elif scenario == "HIGH_VOLATILITY":
            d = rng.normal(mu, sd * 1.5, horizon)
        elif scenario == "SUPPLY_DISRUPTION":
            d = rng.normal(mu, sd, horizon)
            disrupted = rng.choice(horizon, size=int(0.2 * horizon), replace=False)
            d[disrupted] *= 1.3
        elif scenario == "SEASONAL_SURGE":
            d = rng.normal(mu, sd, horizon)
            surges = rng.choice(horizon, size=int(0.25 * horizon), replace=False)
            d[surges] *= rng.uniform(1.8, 2.2, len(surges))
        else:
            raise ValueError(f"Unknown scenario: {scenario}")
        demands[r] = np.maximum(d, 0.0)
    return demands

def generate_unit_costs(n_rep: int, horizon: int, rng: np.random.Generator) -> np.ndarray:
    # Code3-style
    return rng.lognormal(3.5, 0.8, size=(n_rep, horizon)).astype(float)

# -----------------------------------------------------------------------------
# RL training routine for one scenario/forecaster/reward mode
# -----------------------------------------------------------------------------

def train_agent(
    scenario: str,
    demand_train: np.ndarray,
    forecaster: Forecaster,
    cfg: RLConfig,
    cost_params: CostParams,
    route_params: RouteParams,
    reward_mode: str,
    routing_weight: float,
    ga_on_for_reward_proxy: bool,
        seed: int,
    progress: bool = False,
    train_log_every: int = 25,
) -> DoubleQLearningAgent:
    """
    Train on a single trajectory (aligned with Code3) to keep compute bounded.
    Reward depends on reward_mode and includes routing proxy if enabled.
    """
    set_global_seeds(seed)
    rng = np.random.default_rng(seed)

    d = demand_train.astype(float)
    forecaster.fit(d)
    # Build one-step-ahead forecast series for bins
    fc = np.array([forecaster.predict_one(d[:max(1, t)]) for t in range(len(d))], dtype=float)

    agent = DoubleQLearningAgent(cfg)
    agent.init_bins(demand=d, forecast=fc)

    # training episodes
    episodes = 400
    max_t = min(len(d), 365)  # cap
    for ep in range(episodes):
        if progress and (ep == 0 or ((ep + 1) % max(int(train_log_every), 1) == 0) or (ep == episodes - 1)):
            print(f"train_agent intra: scenario={scenario} | reward={reward_mode} | ep={ep+1}/{episodes}")
            sys.stdout.flush()
        inv = float(np.mean(d) * cost_params.safety_stock_factor)
        pipeline: List[Tuple[int, float]] = []
        ep_reward = 0.0

        for t in range(max_t - 1):
            # arrivals
            if pipeline:
                arrived = [q for (ad, q) in pipeline if ad == t]
                if arrived:
                    inv += float(np.sum(arrived))
                pipeline = [(ad, q) for (ad, q) in pipeline if ad != t]

            hist = d[:t]
            f = float(forecaster.predict_one(hist))
            s = agent.discretize(inv, f)
            a = agent.choose_action(s, eval_mode=False)
            order_qty = float(a * agent.order_scale)

            # place order into pipeline
            rc = 0.0
            if order_qty > 0:
                lt = int(lead_time_days(scenario, rng))
                pipeline.append((t + lt, order_qty))
                # routing proxy at ordering time
                rc = routing_cost_proxy(
                    rng=rng, order_qty=order_qty, ga_on=ga_on_for_reward_proxy,
                    route_fixed=route_params.route_fixed, route_per_unit=route_params.route_per_unit
                )

            # demand step
            d_t = float(d[t])
            available = inv
            fulfilled = min(available, d_t)
            lost = max(0.0, d_t - fulfilled)
            inv = max(0.0, available - fulfilled)

            # inventory costs (unit cost = 1.0 for reward scaling)
            stockout_c = lost * cost_params.stockout_penalty + (lost ** 2) * 0.05
            holding_c = inv * cost_params.holding_rate
            ordering_c = order_qty * cost_params.ordering_rate
            service_bonus = cost_params.service_bonus if lost == 0 else 0.0

            total_c = stockout_c + holding_c + ordering_c
            if reward_mode.lower() == "inventory_plus_routing":
                total_c += routing_weight * rc

            reward = -total_c + service_bonus

            # next state
            nf = float(forecaster.predict_one(d[:t+1]))
            ns = agent.discretize(inv, nf)
            agent.update(s, a, reward, ns)
            ep_reward += reward

        agent.decay_epsilon()

    return agent

# -----------------------------------------------------------------------------
# Aggregation / bootstrap CI
# -----------------------------------------------------------------------------

def _fmt_td(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0:
        return "NA"
    td = timedelta(seconds=int(round(seconds)))
    return str(td)

def _progress_line(done: int, total: int, elapsed_s: float) -> str:
    rate = (done / elapsed_s) if elapsed_s > 1e-9 else float("nan")
    eta_s = ((total - done) / rate) if (rate and math.isfinite(rate) and rate > 0) else float("nan")
    pct = 100.0 * done / max(total, 1)
    return f"[{done:,}/{total:,} | {pct:6.2f}%] elapsed={_fmt_td(elapsed_s)} | eta={_fmt_td(eta_s)} | rate={rate:,.2f}/s"

def bootstrap_ci_mean(x: np.ndarray, seed: int = 123, b: int = 2000) -> Tuple[float, float]:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(b, x.size))
    means = x[idx].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--out_dir", type=str, default="reviewer_close_out")
    ap.add_argument("--quick", action="store_true", help="Fewer replications for speed.")
    ap.add_argument("--full", action="store_true", help="More replications (confirmatory).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--progress", action="store_true", help="Print progress and ETA during runs.")
    ap.add_argument("--log_every", type=int, default=250, help="Progress print frequency (in completed policy-episodes).")
    ap.add_argument("--checkpoint_every", type=int, default=5000, help="Write partial raw CSV every N completed policy-episodes (safety).")
    ap.add_argument("--horizons", type=str, default="90,180,365")
    ap.add_argument("--scenarios", type=str, default="STABLE,HIGH_VOLATILITY,SUPPLY_DISRUPTION,SEASONAL_SURGE")
    ap.add_argument("--forecasters", type=str, default="lstm_rf,ses,holt,croston")
    ap.add_argument("--reward_modes", type=str, default="inventory_only,inventory_plus_routing")
    ap.add_argument("--ga_modes", type=str, default="on,off")
    ap.add_argument("--routing_weight", type=float, default=1.0)
    ap.add_argument("--carryover", type=str, default="carryover", choices=["carryover", "truncate"])
    ap.add_argument("--warmup_ratio", type=float, default=0.30)
    ap.add_argument("--data_path", type=str, default="", help="Optional CSV to estimate demand scale.")
    ap.add_argument("--qty_col", type=str, default="", help="Optional: quantity column name for data_path.")
    # ignore notebook args like -f kernel.json
    args, _unknown = ap.parse_known_args(argv)
    return args

def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    # Enable exception telemetry only when progress output is requested.
    # This does not change computations; it only prints warnings if an exception occurs.
    global LOG_EXCEPTIONS
    LOG_EXCEPTIONS = bool(getattr(args, "progress", False))

    # GPU check (verbosity only; does not force GPU usage)
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            tf.config.experimental.set_memory_growth(gpus[0], True)
            if args.progress:
                print('GPU detected:', gpus[0].name)
        else:
            if args.progress:
                print('GPU detected: none')
    except Exception as e:
        if args.progress:
            print('GPU check skipped:', str(e))
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    set_global_seeds(int(args.seed))
    rng_master = np.random.default_rng(int(args.seed))

    horizons = [int(x.strip()) for x in args.horizons.split(",") if x.strip()]
    scenarios = [s.strip().upper() for s in args.scenarios.split(",") if s.strip()]
    forecaster_names = [s.strip().lower() for s in args.forecasters.split(",") if s.strip()]
    reward_modes = [s.strip().lower() for s in args.reward_modes.split(",") if s.strip()]
    ga_modes = [s.strip().lower() for s in args.ga_modes.split(",") if s.strip()]
    routing_weight = float(args.routing_weight)

    if args.full:
        n_rep = 200
    elif args.quick:
        n_rep = 30
    else:
        n_rep = 100

    # Demand scale (baseline or estimated from data)
    dparams = DemandParams(mean=150.0, sd=63.0)
    data_info: Dict[str, str] = {}
    if args.data_path:
        p = Path(args.data_path)
        if p.exists():
            df = pd.read_csv(p)
            qty_col = args.qty_col.strip()
            if not qty_col:
                # best-effort infer numeric quantity column
                numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
                # fall back to common names
                for cand in ["qty", "quantity", "demand", "units", "value"]:
                    if cand in df.columns:
                        qty_col = cand
                        break
                if not qty_col and numeric_cols:
                    qty_col = numeric_cols[0]
            if qty_col and qty_col in df.columns:
                dparams = estimate_params_from_data(df, qty_col=qty_col)
                data_info = {"data_path": str(p), "qty_col": qty_col}
        # if missing, keep defaults

    cost_params = CostParams()
    route_params = RouteParams()

    # Policies (include forecast-parity baseline)
    policies = ["STATIC", "DYNAMIC", "DYN_FC", "BASESTOCK", "AI"]

    total_policy_episodes = (
        len(scenarios) * len(forecaster_names) * len(reward_modes) * len(horizons) * len(ga_modes) * n_rep * len(policies)
    )
    if args.progress:
        print('Planned policy-episodes:', f"{total_policy_episodes:,}")
        print('Replications:', n_rep, '| horizons:', horizons, '| scenarios:', scenarios)
        print('Forecasters:', forecaster_names, '| reward modes:', reward_modes, '| GA modes:', ga_modes)
        print('Carryover:', args.carryover, '| routing_weight:', routing_weight)
        sys.stdout.flush()

    t0 = perf_counter()
    done_policy_episodes = 0
    last_log = 0
    last_heartbeat = t0
    partial_path = out_dir / 'raw_replications_partial.csv'
    if partial_path.exists():
        partial_path.unlink()

    # Manifest
    manifest = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": int(args.seed),
        "n_replications": int(n_rep),
        "horizons": horizons,
        "scenarios": scenarios,
        "forecasters": forecaster_names,
        "reward_modes": reward_modes,
        "ga_modes": ga_modes,
        "carryover": args.carryover,
        "routing_weight": routing_weight,
        "demand_params": {"mean": dparams.mean, "sd": dparams.sd},
        "data_info": data_info,
    }

    # Timing collectors (instrumentation only)
    timing_train_agent: List[Dict[str, object]] = []
    timing_eval_segments: List[Dict[str, object]] = []
    timing_eval_by_block: Dict[str, float] = {}
    t_wall0 = perf_counter()

    rows: List[Dict[str, object]] = []
    # ------------------------------------------------------------------
    # SIGINT (Ctrl+C / Stop) handler: flush partial results and mark run.
    # Telemetry/I-O only; does not affect computations.
    # ------------------------------------------------------------------
    interrupted_flag_path = out_dir / "RUN_INTERRUPTED.txt"

    def _sigint_handler(sig, frame):
        try:
            if rows:
                write_header = (not partial_path.exists())
                pd.DataFrame(rows).to_csv(partial_path, mode="a", index=False, header=write_header)
                rows.clear()
            interrupted_flag_path.write_text(
                f"Interrupted at {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} UTC\n"
                f"done_policy_episodes={done_policy_episodes}\n",
                encoding="utf-8",
            )
            if args.progress:
                print("\nSIGINT: flushed partial CSV and wrote RUN_INTERRUPTED.txt")
                sys.stdout.flush()
        finally:
            raise KeyboardInterrupt

    try:
        signal.signal(signal.SIGINT, _sigint_handler)
    except Exception:
        pass


    # Core experiment loops
    try:
        for s_idx, scenario in enumerate(scenarios, start=1):
            if args.progress:
                print(f"\n=== SCENARIO {s_idx}/{len(scenarios)}: {scenario} ===")
                sys.stdout.flush()
            # pre-generate a long training trajectory for RL per forecaster/reward_mode
            train_h = max(horizons) + int(round(args.warmup_ratio * max(horizons)))
            demand_train = generate_demand(scenario, n_rep=1, horizon=train_h, params=dparams, rng=rng_master)[0]

            for forecaster_name in forecaster_names:
                # fit forecaster and train RL agents per reward_mode
                for reward_mode in reward_modes:
                    forecaster_for_train = make_forecaster(forecaster_name)
                    if args.progress:
                        print(f"train_agent start: scenario={scenario} | forecaster={forecaster_name} | reward={reward_mode}")
                        sys.stdout.flush()
                    t_train0 = perf_counter()
                    agent = train_agent(
                        scenario=scenario,
                        demand_train=demand_train,
                        forecaster=forecaster_for_train,
                        cfg=RLConfig(),
                        cost_params=cost_params,
                        route_params=route_params,
                        reward_mode=reward_mode,
                        routing_weight=routing_weight,
                        ga_on_for_reward_proxy=True,  # proxy assumes GA is available in training cost term
                        seed=int(args.seed) + 7,
                    progress=bool(args.progress),
                    train_log_every=25,
                    )
                    t_train1 = perf_counter()
                    timing_train_agent.append({
                        "scenario": scenario,
                        "forecaster": forecaster_name,
                        "reward_mode": reward_mode,
                        "seconds": float(t_train1 - t_train0),
                    })
                    if args.progress:
                        print(f"train_agent: scenario={scenario} | forecaster={forecaster_name} | reward={reward_mode} | elapsed={_fmt_td(t_train1 - t_train0)}")
                        sys.stdout.flush()

                    for h_idx, H in enumerate(horizons, start=1):
                        block_key = f"{scenario}::H{H}"
                        if args.progress:
                            print(f"\n--- BLOCK {h_idx}/{len(horizons)} | scenario={scenario} | horizon={H} | forecaster={forecaster_name} | reward={reward_mode} ---")
                            sys.stdout.flush()
                        t_eval0 = perf_counter()
                        # Segment bookkeeping: (scenario, forecaster, reward_mode, horizon) across all GA modes, replications, and policies
                        seg_total_episodes = int(len(ga_modes) * n_rep * len(policies))
                        seg_done = 0  # local counter for this block (telemetry)
                        seg_meta = {
                            "scenario": scenario,
                            "horizon_days": int(H),
                            "forecaster": forecaster_name,
                            "reward_mode": reward_mode,
                            "n_replications": int(n_rep),
                            "ga_modes": list(ga_modes),
                            "policies": list(policies),
                        }
                        warmup = int(round(args.warmup_ratio * H))
                        total_len = warmup + H

                        # Generate paired demand and unit costs for this configuration
                        rng_cfg = np.random.default_rng(seed32(args.seed, "cfg", scenario, forecaster_name, reward_mode, H))
                        demand_mat = generate_demand(scenario, n_rep=n_rep, horizon=total_len, params=dparams, rng=rng_cfg)
                        uc_mat = generate_unit_costs(n_rep=n_rep, horizon=total_len, rng=rng_cfg)

                        for ga_mode in ga_modes:
                            ga_on = (ga_mode == "on")

                            for rep in range(n_rep):
                                # CRN: same replication seed for all policies
                                rng_rep = np.random.default_rng(int(args.seed) + 1000 * rep + 13)

                                # Forecaster instance per replication (fit on warm-up)
                                forecaster = make_forecaster(forecaster_name)
                                forecaster.fit(demand_mat[rep, :warmup] if warmup > 0 else demand_mat[rep, :max(14, min(60, total_len))])

                                for policy in policies:
                                    # policy-specific: AI uses trained agent; others ignore agent
                                    metrics = simulate_policy_episode(
                                        scenario=scenario,
                                        demand=demand_mat[rep],
                                        unit_costs=uc_mat[rep],
                                        policy=policy,
                                        forecaster=forecaster,
                                        agent=agent if policy == "AI" else None,
                                        cost_params=cost_params,
                                        route_params=route_params,
                                        ga_on=ga_on,
                                        reward_mode=reward_mode,
                                        routing_weight=routing_weight,
                                        warmup_days=warmup,
                                        eval_days=H,
                                        carryover=args.carryover,
                                        rng=rng_rep,
                                    )
                                    rows.append({
                                        "scenario": scenario,
                                        "horizon_days": H,
                                        "warmup_days": warmup,
                                        "forecaster": forecaster_name,
                                        "reward_mode": reward_mode,
                                        "ga_mode": ga_mode,
                                        "replication": rep,
                                        "policy": policy,
                                        **metrics,
                                    })
                                    # Global progress (instrumentation only)
                                    done_policy_episodes += 1
                                    seg_done += 1
                                    # Periodic checkpoint (I/O only; does not affect results)
                                    if args.checkpoint_every and int(args.checkpoint_every) > 0 and (done_policy_episodes % int(args.checkpoint_every) == 0):
                                        write_header = (not partial_path.exists())
                                        df_ck = pd.DataFrame(rows)
                                        df_ck.to_csv(partial_path, mode="a", index=False, header=write_header)
                                        rows.clear()
                                        if args.progress:
                                            print(f"checkpoint: wrote {len(df_ck):,} rows to {partial_path.name}")
                                            sys.stdout.flush()

                                    # Heartbeat log (time-based; helps when episodes are slow)
                                    if args.progress:
                                        now_hb = perf_counter()
                                        if (now_hb - last_heartbeat) >= 120.0 and (done_policy_episodes != last_log):
                                            elapsed_hb = now_hb - t0
                                            rate_hb = done_policy_episodes / max(elapsed_hb, 1e-9)
                                            remaining_hb = max(total_policy_episodes - done_policy_episodes, 0)
                                            eta_hb_s = remaining_hb / max(rate_hb, 1e-9)
                                            pct_hb = 100.0 * done_policy_episodes / max(total_policy_episodes, 1)
                                            print(f"heartbeat: {done_policy_episodes:,}/{total_policy_episodes:,} ({pct_hb:5.1f}%) | elapsed={_fmt_td(elapsed_hb)} | eta={_fmt_td(eta_hb_s)} | rate={rate_hb:,.2f} ep/s")
                                            sys.stdout.flush()
                                            last_heartbeat = now_hb
                                    if args.progress and (done_policy_episodes - last_log) >= int(args.log_every):
                                        elapsed = perf_counter() - t0
                                        rate = done_policy_episodes / max(elapsed, 1e-9)
                                        remaining = max(total_policy_episodes - done_policy_episodes, 0)
                                        eta_s = remaining / max(rate, 1e-9)
                                        pct = 100.0 * done_policy_episodes / max(total_policy_episodes, 1)
                                        print(f"progress: {done_policy_episodes:,}/{total_policy_episodes:,} ({pct:5.1f}%) | elapsed={_fmt_td(elapsed)} | eta={_fmt_td(eta_s)} | rate={rate:,.2f} ep/s")
                                        sys.stdout.flush()
                                        last_log = done_policy_episodes

                                    # Segment completion (scenario/horizon block)
                                    if seg_done == seg_total_episodes:
                                        # Block-end checkpoint (I/O only; helps recover partial results)
                                        if args.checkpoint_every and int(args.checkpoint_every) > 0 and rows:
                                            write_header = (not partial_path.exists())
                                            df_ck = pd.DataFrame(rows)
                                            df_ck.to_csv(partial_path, mode="a", index=False, header=write_header)
                                            rows.clear()
                                            if args.progress:
                                                print(f"checkpoint(block): wrote {len(df_ck):,} rows to {partial_path.name}")
                                                sys.stdout.flush()
                                        t_eval1 = perf_counter()
                                        seg_elapsed = float(t_eval1 - t_eval0)
                                        timing_eval_segments.append({**seg_meta, "seconds": seg_elapsed})
                                        timing_eval_by_block[block_key] = float(timing_eval_by_block.get(block_key, 0.0) + seg_elapsed)
                                        if args.progress:
                                            print(f"block done: scenario={scenario} | horizon={H} | forecaster={forecaster_name} | reward={reward_mode} | elapsed={_fmt_td(seg_elapsed)}")
                                            sys.stdout.flush()
    except KeyboardInterrupt:
        if args.progress:
            print("\nInterrupted by user. Partial outputs (if any) remain in the output folder.")
            sys.stdout.flush()
        return 130

    if partial_path.exists():
        raw_partial = pd.read_csv(partial_path)
        raw_tail = pd.DataFrame(rows) if rows else pd.DataFrame(columns=raw_partial.columns)
        raw = pd.concat([raw_partial, raw_tail], ignore_index=True)
    else:
        raw = pd.DataFrame(rows)
    raw_path = out_dir / "raw_replications.csv"
    raw.to_csv(raw_path, index=False)

    # Summary means + CI
    summ_rows = []
    for keys, g in raw.groupby(["scenario", "horizon_days", "forecaster", "reward_mode", "ga_mode", "policy"]):
        x = g["total_cost"].to_numpy(dtype=float)
        mean = float(np.mean(x))
        sd = float(np.std(x, ddof=1)) if x.size > 1 else 0.0
        lo, hi = bootstrap_ci_mean(x, seed=123, b=2000)
        summ_rows.append({
            "scenario": keys[0],
            "horizon_days": int(keys[1]),
            "forecaster": keys[2],
            "reward_mode": keys[3],
            "ga_mode": keys[4],
            "policy": keys[5],
            "mean_total_cost": mean,
            "sd_total_cost": sd,
            "ci95_low": lo,
            "ci95_high": hi,
            "n": int(x.size),
        })
    summary = pd.DataFrame(summ_rows).sort_values(
        ["scenario", "horizon_days", "forecaster", "reward_mode", "ga_mode", "mean_total_cost"]
    )
    summary.to_csv(out_dir / "summary_means_ci.csv", index=False)

    # (8) Routing ablation: GA on vs off (routing cost share + delta)
    rout_rows = []
    for (scenario, H, forecaster, reward_mode, policy, rep), g in raw.groupby(
        ["scenario", "horizon_days", "forecaster", "reward_mode", "policy", "replication"]
    ):
        # compare GA on vs off for same rep
        g2 = g.set_index("ga_mode")
        if ("on" in g2.index) and ("off" in g2.index):
            on = g2.loc["on"]
            off = g2.loc["off"]
            rout_rows.append({
                "scenario": scenario, "horizon_days": int(H), "forecaster": forecaster,
                "reward_mode": reward_mode, "policy": policy, "replication": int(rep),
                "routing_share_on": float(on["routing_cost"] / max(on["total_cost"], 1e-9)),
                "routing_share_off": float(off["routing_cost"] / max(off["total_cost"], 1e-9)),
                "delta_total_cost_on_minus_off": float(on["total_cost"] - off["total_cost"]),
                "delta_routing_cost_on_minus_off": float(on["routing_cost"] - off["routing_cost"]),
            })
    rout = pd.DataFrame(rout_rows)
    rout.to_csv(out_dir / "routing_ablation.csv", index=False)

    # (4) Forecasting ablation: ranking stability across forecasters (per scenario/horizon/reward/ga)
    rank_rows = []
    for (scenario, H, reward_mode, ga_mode), g in summary.groupby(["scenario", "horizon_days", "reward_mode", "ga_mode"]):
        # rank policies by mean cost within each forecaster
        per_fc = {}
        for fc, gg in g.groupby("forecaster"):
            r = gg.sort_values("mean_total_cost")[["policy", "mean_total_cost"]].reset_index(drop=True)
            per_fc[fc] = list(r["policy"].values)

        # compute pairwise Kendall tau on ranks (simple implementation)
        fcs = sorted(per_fc.keys())
        for i in range(len(fcs)):
            for j in range(i + 1, len(fcs)):
                a, b = fcs[i], fcs[j]
                order_a = per_fc[a]
                order_b = per_fc[b]
                # map to ranks
                ra = {p: k for k, p in enumerate(order_a)}
                rb = {p: k for k, p in enumerate(order_b)}
                policies_all = [p for p in order_a if p in rb]
                # kendall tau (no ties expected)
                conc, disc = 0, 0
                for u in range(len(policies_all)):
                    for v in range(u + 1, len(policies_all)):
                        pu, pv = policies_all[u], policies_all[v]
                        s1 = ra[pu] - ra[pv]
                        s2 = rb[pu] - rb[pv]
                        if s1 * s2 > 0:
                            conc += 1
                        else:
                            disc += 1
                denom = conc + disc
                tau = (conc - disc) / denom if denom > 0 else float("nan")
                rank_rows.append({
                    "scenario": scenario, "horizon_days": int(H), "reward_mode": reward_mode,
                    "ga_mode": ga_mode, "forecaster_a": a, "forecaster_b": b,
                    "kendall_tau_policy_ranking": float(tau),
                })
    pd.DataFrame(rank_rows).to_csv(out_dir / "forecast_ranking_stability.csv", index=False)

    # Wilcoxon + Holm (paired, within each scenario/horizon/forecaster/reward/ga) comparing AI vs each baseline
    tests = []
    for (scenario, H, forecaster, reward_mode, ga_mode), g in raw.groupby(
        ["scenario", "horizon_days", "forecaster", "reward_mode", "ga_mode"]
    ):
        # pivot replication x policy
        piv = g.pivot_table(index="replication", columns="policy", values="total_cost", aggfunc="mean")
        if "AI" not in piv.columns:
            continue
        baselines = [c for c in piv.columns if c != "AI"]
        pvals = []
        labels = []
        for bsl in baselines:
            x = piv["AI"].to_numpy(dtype=float)
            y = piv[bsl].to_numpy(dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() < 5:
                continue
            stat = wilcoxon(x[mask], y[mask], alternative="two-sided", zero_method="wilcox")
            pvals.append(float(stat.pvalue))
            labels.append(bsl)
        if not pvals:
            continue
        adj = holm_adjust(pvals)
        for bsl, p, p_adj in zip(labels, pvals, adj):
            tests.append({
                "scenario": scenario,
                "horizon_days": int(H),
                "forecaster": forecaster,
                "reward_mode": reward_mode,
                "ga_mode": ga_mode,
                "comparison": f"AI vs {bsl}",
                "p_value": float(p),
                "p_value_holm": float(p_adj),
                "n_pairs": int(piv.shape[0]),
            })
    pd.DataFrame(tests).to_csv(out_dir / "wilcoxon_holm.csv", index=False)

    # Finalize run manifest with instrumentation timings
    t_wall1 = perf_counter()
    timings = {
        "wallclock_seconds": float(t_wall1 - t_wall0),
        "train_agent": timing_train_agent,
        "evaluation_segments": timing_eval_segments,
        "evaluation_by_block_seconds": timing_eval_by_block,
    }
    manifest["timings"] = timings
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Telemetry exports (CSV) for spreadsheet review
    if timing_train_agent:
        pd.DataFrame(timing_train_agent).to_csv(out_dir / "timings_train_agent.csv", index=False)
    if timing_eval_segments:
        pd.DataFrame(timing_eval_segments).to_csv(out_dir / "timings_eval_segments.csv", index=False)
    if timing_eval_by_block:
        pd.DataFrame([{"block": k, "seconds": v} for k, v in timing_eval_by_block.items()]).to_csv(
            out_dir / "timings_by_block.csv", index=False
        )

    # Reproducibility fingerprint (hashes) for the raw and primary CSV outputs.
    # Telemetry only: does not affect simulation results.
    primary_paths = [
        raw_path,
        out_dir / "summary_means_ci.csv",
        out_dir / "wilcoxon_holm.csv",
        out_dir / "routing_ablation.csv",
        out_dir / "forecast_ranking_stability.csv",
        out_dir / "run_manifest.json",
    ]
    # Optional timing CSVs (present only if instrumentation was enabled)
    for opt in [
        out_dir / "timings_train_agent.csv",
        out_dir / "timings_eval_segments.csv",
        out_dir / "timings_by_block.csv",
    ]:
        if opt.exists():
            primary_paths.append(opt)

    fp = {
        "raw_path": str(raw_path),
        "raw_sha256": _sha256_file(raw_path) if raw_path.exists() else None,
        "outputs": {p.name: _sha256_file(p) for p in primary_paths if p.exists() and p.is_file()},
        "environment": {
            "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        },
        "key_scalars": {
            "n_rows_raw": int(len(raw)),
            "n_scenarios": int(raw["scenario"].nunique()),
            "n_policies": int(raw["policy"].nunique()),
            "overall_mean_total_cost": float(np.mean(raw["total_cost"].to_numpy(dtype=float))),
        },
    }
    fp_path = out_dir / "run_fingerprint.json"
    fp_path.write_text(json.dumps(fp, indent=2, sort_keys=True), encoding="utf-8")


    print(f"Saved:")
    print(f"  - {raw_path}")
    print(f"  - {out_dir / 'summary_means_ci.csv'}")
    print(f"  - {out_dir / 'wilcoxon_holm.csv'}")
    print(f"  - {out_dir / 'routing_ablation.csv'}")
    print(f"  - {out_dir / 'forecast_ranking_stability.csv'}")
    print(f"  - {out_dir / 'run_fingerprint.json'}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
