import os
import pandas as pd
import numpy as np
import random
import logging
import warnings
import time
import gc
import sys
from collections import defaultdict
from scipy.stats import mannwhitneyu, levene, t
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from skopt import gp_minimize
from skopt.space import Real, Integer
import tensorflow as tf
from tensorflow.keras import Input, Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from deap import base, creator, tools, algorithms
from statsmodels.stats.power import TTestIndPower

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

NUM_MONTE_CARLO = 100
CUTOFF_DATE = '2021-06-30'
OUTPUT_DIR = 'outputs'
if not os.path.exists(OUTPUT_DIR): 
    os.makedirs(OUTPUT_DIR)

HORIZON_DAYS = 90
HYPEROPT_AVAILABLE = True
HYPEROPT_N_CALLS = 10
HYPEROPT_N_INIT = 10
BOOTSTRAP_SAMPLES = 1000
LSTM_EPOCHS = 12
BONFERRONI_ALPHA = 0.017
NUM_COMPARISONS = 6

CORRECTED_PARAMS = {
    'stockout_penalty': 25.0,
    'holding_rate': 0.003,
    'ordering_rate': 1.5,
    'service_bonus': 8.0,
    'target_fill_rate': 0.97,
    'safety_stock_factor': 2.0
}

if not hasattr(creator, 'FitnessMin'):
    creator.create('FitnessMin', base.Fitness, weights=(-1.0,))
if not hasattr(creator, 'Individual'):
    creator.create('Individual', list, fitness=creator.FitnessMin)

def optimized_cleanup():
    try:
        tf.keras.backend.clear_session()
    except Exception:
        pass
    gc.collect()

class DataProcessor:
    def __init__(self, data_path='data/healthcare_supply_chain.csv'):
        self.data_path = data_path
        
    def process(self):
        print(f"[DataProcessor] Loading data from {self.data_path}...", flush=True)
        
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(
                f"Data file not found: {self.data_path}\n"
                f"Please provide a CSV file with the following columns:\n"
                f"DATA, CODIGO, ALMOXARIFADO, QUANTIDADE, UNITCOST, LATITUDE, LONGITUDE"
            )
        
        df = pd.read_csv(self.data_path)
        print(f"[DataProcessor] Raw records: {len(df)}", flush=True)
        
        df['DATA'] = pd.to_datetime(df['DATA'], errors='coerce')
        df = df.dropna(subset=['DATA', 'ALMOXARIFADO', 'CODIGO', 'QUANTIDADE'])
        df = df[df['QUANTIDADE'] > 0]
        
        if 'UNITCOST' not in df.columns:
            if 'VALOR' in df.columns:
                df['UNITCOST'] = df['VALOR'] / df['QUANTIDADE']
            else:
                df['UNITCOST'] = np.random.lognormal(3.3, 0.7, len(df))
        
        df = df.dropna(subset=['UNITCOST'])
        df = df[df['UNITCOST'] <= df['UNITCOST'].quantile(0.99)]
        
        coords = df.groupby('ALMOXARIFADO').size().reset_index()[['ALMOXARIFADO']]
        
        if 'LATITUDE' in df.columns and 'LONGITUDE' in df.columns:
            lat_lon = df.groupby('ALMOXARIFADO')[['LATITUDE', 'LONGITUDE']].first().reset_index()
            coords = coords.merge(lat_lon, on='ALMOXARIFADO')
        else:
            coords['LATITUDE'] = np.random.uniform(-24, -3, len(coords))
            coords['LONGITUDE'] = np.random.uniform(-57, -35, len(coords))
        
        df = df.merge(coords, on='ALMOXARIFADO', how='left', suffixes=('', '_coord'))
        
        if 'LATITUDE_coord' in df.columns:
            df['LATITUDE'] = df['LATITUDE_coord']
            df['LONGITUDE'] = df['LONGITUDE_coord']
            df = df.drop(['LATITUDE_coord', 'LONGITUDE_coord'], axis=1)
        
        g = df.groupby(['DATA', 'ALMOXARIFADO', 'CODIGO', 'LATITUDE', 'LONGITUDE'], as_index=False).agg({
            'QUANTIDADE': 'sum',
            'UNITCOST': 'mean'
        })
        
        print(f"[DataProcessor] Processed records: {len(g)}", flush=True)
        return g, coords

class PerformanceMonitor:
    def __init__(self, patience=6, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.wait = 0
        
    def check_convergence(self, current_loss):
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.wait = 0
            return False
        else:
            self.wait += 1
            return self.wait >= self.patience

class CheckpointManager:
    def __init__(self, filepath='checkpoint.csv'):
        self.filepath = os.path.join(OUTPUT_DIR, filepath)
        
    def save(self, results_df):
        results_df.to_csv(self.filepath, index=False)
        print(f"[Checkpoint] Saved: {self.filepath} ({len(results_df)} records)", flush=True)

class ProgressLogger:
    def __init__(self, total_units):
        self.total_units = total_units
        self.start_time = time.time()
        self.unit_times = []
        
    def log_unit_start(self, unit_idx, unit_name):
        if unit_idx > 0:
            elapsed_time = time.time() - self.start_time
            avg_time_per_unit = elapsed_time / unit_idx
            self.unit_times.append(avg_time_per_unit)
            remaining_units = self.total_units - unit_idx
            estimated_remaining_sec = np.mean(self.unit_times) * remaining_units
            print(f"[Progress] Estimated time remaining: {estimated_remaining_sec / 60:.1f} minutes", flush=True)
        print(f"[Progress] Processing unit {unit_idx + 1}/{self.total_units}: {unit_name}", flush=True)

class LSTMRFEnsemble:
    def __init__(self, seq_len=14, lstm_units=64, dropout_rate=0.3):
        self.seq_len = seq_len
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.lstm = None
        self.rf = None
        self.rf_params = {'n_estimators': 120, 'max_depth': 12}
        self.is_fitted = False
        self.fallback = 0
        
    def _seq(self, data):
        X, y = [], []
        for i in range(self.seq_len, len(data)):
            X.append(data[i-self.seq_len:i])
            y.append(data[i])
        return np.array(X), np.array(y)
    
    def fit(self, df):
        data = df['QUANTIDADE'].values.astype(float)
        self.fallback = data.mean() if len(data) > 0 else 0.0
        
        if len(data) < self.seq_len * 3:
            self.lstm = None
            self.rf = RandomForestRegressor(**self.rf_params, min_samples_split=5, random_state=42)
            X = np.array([data[max(0, i-self.seq_len):i] for i in range(self.seq_len, len(data))])
            y = data[self.seq_len:] if len(data) > self.seq_len else data
            if len(X) > 0 and len(y) == len(X):
                self.rf.fit(X, y)
                self.is_fitted = True
            return
        
        X, y = self._seq(data)
        X_scaled = self.scaler_X.fit_transform(X.reshape(-1, self.seq_len)).reshape(-1, self.seq_len, 1)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
        
        idx = int(0.8 * len(X))
        Xtr_lstm, Xv_lstm, ytr_lstm, yv_lstm = X_scaled[:idx], X_scaled[idx:], y_scaled[:idx], y_scaled[idx:]
        Xtr_rf, Xv_rf, ytr_rf, yv_rf = X[:idx], X[idx:], y[:idx], y[idx:]
        
        self.lstm = Sequential([
            Input(shape=(self.seq_len, 1)),
            LSTM(self.lstm_units, return_sequences=False),
            Dropout(self.dropout_rate),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        self.lstm.compile(optimizer='adam', loss='mse')
        
        es = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        self.lstm.fit(Xtr_lstm, ytr_lstm, validation_data=(Xv_lstm, yv_lstm), 
                     epochs=LSTM_EPOCHS, batch_size=8, callbacks=[es], verbose=0)
        
        self.rf = RandomForestRegressor(**self.rf_params, min_samples_split=5, random_state=42)
        self.rf.fit(Xtr_rf, ytr_rf)
        self.is_fitted = True
    
    def predict(self, df):
        if not self.is_fitted or len(df) < self.seq_len:
            return self.fallback
        
        seq = df['QUANTIDADE'].values[-self.seq_len:].astype(float)
        preds = []
        
        if self.lstm is not None:
            try:
                seq_scaled = self.scaler_X.transform(seq.reshape(1, -1)).reshape(1, self.seq_len, 1)
                lstm_pred = self.lstm.predict(seq_scaled, verbose=0)[0, 0]
                lstm_pred_unscaled = self.scaler_y.inverse_transform(np.array([[lstm_pred]]))[0, 0]
                preds.append(float(lstm_pred_unscaled))
            except Exception:
                pass
        
        try:
            rf_pred = self.rf.predict(seq.reshape(1, -1))[0]
            preds.append(float(rf_pred))
        except Exception:
            pass
        
        return max(0.0, float(np.mean(preds))) if preds else self.fallback

class QLearningAgent:
    def __init__(self, n_states=100, n_actions=5):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = 0.1
        self.gamma = 0.95
        self.epsilon = 0.2
        self.epsilon_min = 0.02
        self.q1 = np.zeros((n_states, n_actions))
        self.q2 = np.zeros((n_states, n_actions))
        self.order_scale = 1.0
        self.state_bins = None
        self.historical_data = []
        
    def build_bins(self, demand, forecast):
        self.state_bins = {
            'stock': np.percentile(demand * 3, np.linspace(0, 100, 11)),
            'forecast': np.percentile(forecast, np.linspace(0, 100, 11))
        }
        self.order_scale = max(1.0, np.percentile(demand, 90) / (self.n_actions - 1))
        self.historical_data = demand.tolist()
    
    def discretize(self, stock, forecast):
        if self.state_bins is None:
            return 0
        sb = np.clip(np.digitize(stock, self.state_bins['stock']) - 1, 0, 9)
        fb = np.clip(np.digitize(forecast, self.state_bins['forecast']) - 1, 0, 9)
        return min(self.n_states - 1, sb * 10 + fb)
    
    def choose(self, state, eval_mode=False):
        state = int(np.clip(state, 0, self.n_states - 1))
        if eval_mode or random.random() >= self.epsilon:
            return int(np.argmax((self.q1[state] + self.q2[state]) / 2))
        return random.randint(0, self.n_actions - 1)
    
    def calculate_reward(self, inv_before, demand, order, unit_cost):
        inv_after = inv_before + order
        stockout = max(0.0, demand - inv_after)
        excess = max(0.0, inv_after - demand)
        penal = stockout * unit_cost * CORRECTED_PARAMS['stockout_penalty'] + (stockout ** 2) * 0.05
        cost = penal + excess * unit_cost * CORRECTED_PARAMS['holding_rate'] + order * CORRECTED_PARAMS['ordering_rate']
        bonus = CORRECTED_PARAMS['service_bonus'] * unit_cost if stockout == 0 else 0.0
        return -(cost) + bonus, max(0.0, inv_after - demand), (stockout == 0)
    
    def update(self, s, a, r, ns):
        s, a, ns = int(np.clip(s, 0, self.n_states - 1)), int(np.clip(a, 0, self.n_actions - 1)), int(np.clip(ns, 0, self.n_states - 1))
        if random.random() < 0.5:
            na = np.argmax(self.q1[ns])
            target = r + self.gamma * self.q2[ns, na]
            self.q1[s, a] += self.alpha * (target - self.q1[s, a])
        else:
            na = np.argmax(self.q2[ns])
            target = r + self.gamma * self.q1[ns, na]
            self.q2[s, a] += self.alpha * (target - self.q2[s, a])
    
    def train(self, df, forecast_model):
        demand = df['QUANTIDADE'].values.astype(float)
        cost = df['UNITCOST'].values.astype(float)
        forecasts = demand.copy()
        self.build_bins(demand, forecasts)
        
        ep_schedule = [30, 40, 30]
        monitor = PerformanceMonitor(patience=10)
        
        for ep_count in ep_schedule:
            for ep in range(ep_count):
                inv = demand.mean() * CORRECTED_PARAMS['safety_stock_factor']
                ep_rewards = []
                for t in range(len(demand) - 1):
                    s = self.discretize(inv, forecasts[t])
                    a = self.choose(s)
                    order = a * self.order_scale
                    r, inv, _ = self.calculate_reward(inv, demand[t], order, cost[t])
                    ns = self.discretize(inv, forecasts[t + 1])
                    self.update(s, a, r, ns)
                    ep_rewards.append(r)
                self.epsilon = max(self.epsilon_min, self.epsilon * 0.995)
                mean_reward = np.mean(ep_rewards)
                if monitor.check_convergence(-mean_reward):
                    break
            if monitor.wait >= monitor.patience:
                break

def _valid_point(loc):
    if isinstance(loc, (tuple, list)) and len(loc) == 2 and all(np.isscalar(x) for x in loc):
        lat, lon = float(loc[0]), float(loc[1])
        if not (np.isnan(lat) or np.isnan(lon)):
            return True
    return False

def run_deap_vrp_hybrid(locations, ngen=12, timeout_sec=1.5, max_points=7):
    start_time = time.time()
    monitor = PerformanceMonitor(patience=3)
    
    if not locations or len(locations) < 2:
        return 0.0
    
    points = [tuple(map(float, loc)) for loc in locations if _valid_point(loc)]
    if len(points) < 2:
        return 0.0
    
    if len(points) > max_points:
        points = random.sample(points, max_points)
    
    dist_matrix = np.zeros((len(points), len(points)))
    for i in range(len(points)):
        for j in range(len(points)):
            dist_matrix[i, j] = np.sqrt((points[i][0] - points[j][0]) ** 2 + (points[i][1] - points[j][1]) ** 2)
    
    toolbox = base.Toolbox()
    toolbox.register("indices", random.sample, range(len(points)), len(points))
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    def eval_route(ind):
        total = 0.0
        for i in range(len(ind)):
            total += float(dist_matrix[ind[i - 1], ind[i]])
        return (total,)
    
    toolbox.register("evaluate", eval_route)
    toolbox.register("mate", tools.cxOrdered)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    pop = toolbox.population(n=30)
    hof = tools.HallOfFame(1)
    
    for gen in range(ngen):
        if time.time() - start_time > timeout_sec:
            break
        pop, _ = algorithms.eaSimple(pop, toolbox, cxpb=0.8, mutpb=0.2, ngen=1, halloffame=hof, verbose=False)
        best = hof[0].fitness.values[0] if len(hof) > 0 else np.inf
        if monitor.check_convergence(best):
            break
    
    if len(hof) == 0:
        return 0.0
    return float(hof[0].fitness.values[0]) * 0.5

class SupplyChainOptimizer:
    def __init__(self):
        self.p = CORRECTED_PARAMS
    
    def _calculate_period_cost(self, inv, d, order, uc, locations, vrp_prob=0.3):
        inv_after = inv + order
        stockout = max(0.0, d - inv_after)
        excess = max(0.0, inv_after - d)
        sc = stockout * uc * self.p['stockout_penalty'] + (stockout ** 2) * 0.05
        hc = excess * uc * self.p['holding_rate']
        oc = order * self.p['ordering_rate']
        tc = 0.0
        if order > 0 and random.random() < vrp_prob:
            tc = run_deap_vrp_hybrid(locations, ngen=12, timeout_sec=1.5, max_points=7)
        return max(0.0, inv_after - d), sc + hc + oc + tc, (stockout == 0)
    
    def static(self, demand, cost, locations):
        avg, sd = float(np.mean(demand)), float(np.std(demand))
        inv = avg * self.p['safety_stock_factor']
        total_cost, served = 0.0, []
        for d, uc in zip(demand, cost):
            rp = avg + sd * 1.65
            order = max(avg * 1.2, rp - inv) if inv <= rp else 0.0
            inv, c, s = self._calculate_period_cost(inv, d, order, uc, locations)
            total_cost += c
            served.append(s)
        return total_cost / HORIZON_DAYS, float(np.mean(served)) if served else 1.0
    
    def dynamic(self, demand, cost, locations):
        inv = float(np.mean(demand)) * self.p['safety_stock_factor']
        total_cost, served = 0.0, []
        for i, (d, uc) in enumerate(zip(demand, cost)):
            hist = demand[max(0, i - 4):i + 1]
            ma = float(np.mean(hist))
            ms = float(np.std(hist)) if len(hist) > 1 else float(d) * 0.25
            rp = ma + ms * 1.8
            order = max(ma * 1.1, rp - inv) if inv <= rp else 0.0
            inv, c, s = self._calculate_period_cost(inv, d, order, uc, locations)
            total_cost += c
            served.append(s)
        return total_cost / HORIZON_DAYS, float(np.mean(served)) if served else 1.0
    
    def ai(self, demand, cost, forecast, agent, locations):
        inv = float(np.mean(demand)) * self.p['safety_stock_factor']
        total_cost, served = 0.0, []
        for d, uc, f in zip(demand, cost, forecast):
            s = agent.discretize(inv, f)
            a = agent.choose(s, eval_mode=True)
            order = a * agent.order_scale
            inv, c, srv = self._calculate_period_cost(inv, d, order, uc, locations)
            total_cost += c
            served.append(srv)
        return total_cost / HORIZON_DAYS, float(np.mean(served)) if served else 1.0
    
    def base_stock_perfect(self, demand, cost, locations, lead_time=5):
        z_score = 1.96
        demand_lt_mean = float(np.mean(demand)) * lead_time
        demand_lt_std = float(np.std(demand)) * np.sqrt(lead_time)
        base_stock_level = demand_lt_mean + z_score * demand_lt_std
        inventory, total_cost, served = base_stock_level, 0.0, []
        for d, uc in zip(demand, cost):
            order = max(0.0, base_stock_level - inventory)
            inventory, c, s = self._calculate_period_cost(inventory, d, order, uc, locations, vrp_prob=0.3)
            total_cost += c
            served.append(s)
        return total_cost / HORIZON_DAYS, float(np.mean(served)) if served else 1.0
    
    def base_stock_uncertain(self, demand, cost, forecast, locations, lead_time=5):
        z_score = 1.96
        forecast_mean = float(np.mean(forecast))
        forecast_std = float(np.std(forecast))
        demand_lt_mean = forecast_mean * lead_time
        demand_lt_std = forecast_std * np.sqrt(lead_time)
        base_stock_level = demand_lt_mean + z_score * demand_lt_std
        inventory, total_cost, served = base_stock_level, 0.0, []
        for d, uc in zip(demand, cost):
            order = max(0.0, base_stock_level - inventory)
            inventory, c, s = self._calculate_period_cost(inventory, d, order, uc, locations, vrp_prob=0.3)
            total_cost += c
            served.append(s)
        return total_cost / HORIZON_DAYS, float(np.mean(served)) if served else 1.0

def glass_delta(group1, group2):
    n1, n2 = len(group1), len(group2)
    var1 = np.var(group1, ddof=1)
    var2 = np.var(group2, ddof=1)
    pooled_sd = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return (np.mean(group1) - np.mean(group2)) / pooled_sd if pooled_sd > 0 else 0.0

def bootstrap_ci(arr, n=BOOTSTRAP_SAMPLES, alpha=0.05):
    means = [np.random.choice(arr, len(arr), True).mean() for _ in range(n)]
    return np.percentile(means, [alpha / 2 * 100, (1 - alpha / 2) * 100]).tolist()

class SimulationSystem:
    def __init__(self, data_path='data/healthcare_supply_chain.csv'):
        self.data_processor = DataProcessor(data_path=data_path)
        self.optimizer = SupplyChainOptimizer()
        self.checkpoint_manager = CheckpointManager()
    
    def run(self):
        print("\n" + "=" * 80, flush=True)
        print("HYBRID AI-RL-GA FRAMEWORK FOR HEALTHCARE SUPPLY CHAIN OPTIMIZATION", flush=True)
        print("=" * 80, flush=True)
        t0 = time.time()
        
        data, coords = self.data_processor.process()
        if data is None or len(data) == 0:
            print("[Error] Data processing failed", flush=True)
            return
        
        units = data['ALMOXARIFADO'].unique()
        print(f"[SimulationSystem] Units to process: {list(units)}", flush=True)
        
        results_list = []
        progress = ProgressLogger(len(units))
        
        for unit_idx, unit in enumerate(units):
            progress.log_unit_start(unit_idx, unit)
            unit_data = data[data['ALMOXARIFADO'] == unit].sort_values('DATA')
            
            cutoff = pd.to_datetime(CUTOFF_DATE)
            train = unit_data[unit_data['DATA'] <= cutoff]
            test = unit_data[unit_data['DATA'] > cutoff]
            
            print(f"[{unit}] Train: {len(train)}, Test: {len(test)}", flush=True)
            
            if len(train) < 28 or len(test) < 8:
                print(f"[{unit}] Insufficient data, skipping...", flush=True)
                continue
            
            print(f"[{unit}] Training LSTM-RF ensemble...", flush=True)
            ensemble = LSTMRFEnsemble(seq_len=14)
            ensemble.fit(train)
            optimized_cleanup()
            
            print(f"[{unit}] Training RL agent (100 episodes)...", flush=True)
            agent = QLearningAgent(n_actions=5)
            agent.train(train, ensemble)
            
            print(f"[{unit}] Generating ML forecasts...", flush=True)
            forecasts = []
            for i in range(len(test)):
                hist = pd.concat([train, test.iloc[:i + 1]])
                forecasts.append(ensemble.predict(hist))
            
            locs = list(zip(unit_data['LATITUDE'].values, unit_data['LONGITUDE'].values))[:10]
            base_demand = test['QUANTIDADE'].values.astype(float)
            base_cost = test['UNITCOST'].values.astype(float)
            
            print(f"[{unit}] Running {NUM_MONTE_CARLO} Monte Carlo simulations (5 policies)...", flush=True)
            for sim in range(NUM_MONTE_CARLO):
                d_pert = base_demand * np.random.lognormal(0, 0.15, len(base_demand))
                c_pert = base_cost * np.abs(np.random.normal(1, 0.1, len(base_cost)))
                f_pert = np.array(forecasts) * np.abs(np.random.normal(1, 0.2, len(forecasts)))
                
                sc, sfr = self.optimizer.static(d_pert, c_pert, locs)
                dc, dfr = self.optimizer.dynamic(d_pert, c_pert, locs)
                ac, afr = self.optimizer.ai(d_pert, c_pert, f_pert, agent, locs)
                bpc, bpfr = self.optimizer.base_stock_perfect(d_pert, c_pert, locs)
                buc, bufr = self.optimizer.base_stock_uncertain(d_pert, c_pert, f_pert, locs)
                
                results_list.append({
                    'unit': unit, 'sim': sim,
                    'static_cost': sc, 'dynamic_cost': dc, 'ai_cost': ac,
                    'base_stock_perfect_cost': bpc, 'base_stock_uncertain_cost': buc,
                    'static_fr': sfr, 'dynamic_fr': dfr, 'ai_fr': afr,
                    'base_stock_perfect_fr': bpfr, 'base_stock_uncertain_fr': bufr
                })
                
                if (sim + 1) % max(1, NUM_MONTE_CARLO // 5) == 0:
                    print(f"[{unit}] Simulation {sim + 1}/{NUM_MONTE_CARLO} completed", flush=True)
            
            optimized_cleanup()
            print(f"[{unit}] Completed", flush=True)
        
        results_df = pd.DataFrame(results_list)
        self.checkpoint_manager.save(results_df)
        
        if len(results_df) > 0:
            self._analyze_results(results_df)
        
        t1 = time.time()
        print("\n" + "=" * 80, flush=True)
        print(f"SIMULATION COMPLETED in {(t1 - t0) / 60:.1f} minutes", flush=True)
        print("=" * 80 + "\n", flush=True)
    
    def _analyze_results(self, results_df):
        print("\n[STATISTICAL ANALYSIS - Mann-Whitney U with Bonferroni Correction]", flush=True)
        print("[Costs normalized by temporal horizon (90 days)]", flush=True)
        print("\n" + "=" * 80, flush=True)
        print("DESCRIPTIVE STATISTICS (Normalized cost in currency units):", flush=True)
        print("=" * 80, flush=True)
        
        scenarios = ['static', 'dynamic', 'ai', 'base_stock_perfect', 'base_stock_uncertain']
        scenario_data = {}
        
        for sc in scenarios:
            costs = results_df[f'{sc}_cost'].values.astype(float)
            frs = results_df[f'{sc}_fr'].values.astype(float)
            ci = bootstrap_ci(costs)
            scenario_data[sc] = {'costs': costs, 'frs': frs}
            print(f"{sc:25s}: Cost={costs.mean():>15,.2f}±{costs.std():>15,.2f} "
                  f"CI95%=[{ci[0]:>15,.2f},{ci[1]:>15,.2f}] FR={frs.mean():.4f}", flush=True)
        
        print("\n" + "=" * 80, flush=True)
        print("STATISTICAL TESTS (Mann-Whitney U + Glass's Delta + Bonferroni):", flush=True)
        print("=" * 80, flush=True)
        
        pairs = [
            ('dynamic', 'static'),
            ('ai', 'static'),
            ('ai', 'dynamic'),
            ('base_stock_perfect', 'static'),
            ('base_stock_uncertain', 'ai'),
            ('base_stock_uncertain', 'base_stock_perfect')
        ]
        
        bonf_alpha = BONFERRONI_ALPHA / len(pairs)
        stats_results = []
        
        for a, b in pairs:
            a_costs = scenario_data[a]['costs']
            b_costs = scenario_data[b]['costs']
            u, p = mannwhitneyu(a_costs, b_costs, alternative='two-sided')
            delta = glass_delta(a_costs, b_costs)
            sig = "✓ SIG" if p < bonf_alpha else "  "
            print(f"{a:25s} vs {b:25s}: U={u:>8.0f} p={p:.6f} {sig} Glass's Δ={delta:>8.4f}", flush=True)
            stats_results.append({
                'comparison': f"{a}_vs_{b}",
                'u_stat': u,
                'p_value': p,
                'bonf_sig': p < bonf_alpha,
                'glass_delta': delta
            })
        
        stats_df = pd.DataFrame(stats_results)
        stats_df.to_csv(os.path.join(OUTPUT_DIR, 'statistical_tests.csv'), index=False)
        
        summary_df = pd.DataFrame([{
            'policy': sc,
            'mean_cost': scenario_data[sc]['costs'].mean(),
            'std_cost': scenario_data[sc]['costs'].std(),
            'mean_fr': scenario_data[sc]['frs'].mean()
        } for sc in scenarios])
        summary_df.to_csv(os.path.join(OUTPUT_DIR, 'results_summary.csv'), index=False)
        
        print(f"\n[Results] Statistical file: {os.path.join(OUTPUT_DIR, 'statistical_tests.csv')}", flush=True)
        print(f"[Results] Summary file: {os.path.join(OUTPUT_DIR, 'results_summary.csv')}", flush=True)

if __name__ == "__main__":
    data_path = os.environ.get('DATA_PATH', 'data/healthcare_supply_chain.csv')
    SimulationSystem(data_path=data_path).run()
