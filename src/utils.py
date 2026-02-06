import polars as pl
import numpy as np
import json
import datetime
from scipy.interpolate import PchipInterpolator
import os
import scipy.linalg as linalg

def robust_read_csv(path):
    if not os.path.exists(path): raise FileNotFoundError(f"File not found: {path}")
    sep = ','
    try:
        with open(path, 'r') as f:
            header = f.readline()
            if ';' in header: sep = ';'
            elif '\t' in header: sep = '\t'
    except: 
        pass
    try:
        df = pl.read_csv(path, separator=sep, ignore_errors=True, infer_schema_length=10000)
    except:
        df = pl.read_csv(path, separator=sep, ignore_errors=True)
        
    # strip whitespace, upper case, remove quotes
    new_cols = [str(c).strip().strip('[]"\'').upper() for c in df.columns]
    df = df.rename(dict(zip(df.columns, new_cols)))
    return df

def load_curve(csv_path, name="Curve"):
    """ convert input discrete discout factors in interpolated yied curves """

    print(f"Loading {name}: {csv_path}")
    df = robust_read_csv(csv_path)
    
    col_t = next((c for c in df.columns if 'TICKER' in c or 'TENOR' in c or 'T' == c), None)
    col_v = next((c for c in df.columns if 'DF' in c or 'VALUE' in c or 'RATE' in c), None)
    
    if not col_t or not col_v: raise ValueError(f"{name} CSV missing columns.")
    
    def parse_ticker(val):
        s = str(val).upper().strip()
        if '/' in s: 
            s = s.split('/')[-1]
        try:
            if s.endswith('Y'): return float(s[:-1])
            if s.endswith('M'): return float(s[:-1])/12.0
        except: 
            return None
        return None

    df = df.with_columns(
        pl.col(col_t).map_elements(parse_ticker, return_dtype=pl.Float64).alias('T'),
        pl.col(col_v).cast(pl.Float64, strict=False).alias('DF')
    )
    
    # drop nulls and sort
    df = df.drop_nulls(subset=['T', 'DF']).sort('T')
    
    # add T=0 row if missing
    min_t = df['T'].min()
    if min_t is not None and min_t > 0.001:
        row_0 = pl.DataFrame({'T': [0.0], 'DF': [1.0]})
        # select only needed columns to ensure concat works
        df = df.select(['T', 'DF'])
        df = pl.concat([row_0, df], how="vertical")
    else:
        df = df.select(['T', 'DF'])

    # interpolation (PCHIP)

    return PchipInterpolator(df['T'].to_numpy(), np.log(df['DF'].to_numpy()), extrapolate=True)

def get_fwd_rates(interp_func, grid_tau=0.5, max_maturity=10.0):
    """ (1/tau) * (P1/P2 -1) """

    grid = np.arange(0, max_maturity + grid_tau + 0.001, grid_tau)
    # interpolate disc factors 
    dfs = np.exp(interp_func(grid))
    # (1/tau) * (D1/D2 -1)
    fwds = (dfs[:-1] / dfs[1:] - 1) / grid_tau
    return grid, fwds

def get_sabr_params(vol_file, target_tenor, calib_func):
    # handles JSON logic
    root_dir = os.getcwd() 
    log_dir = os.path.join(root_dir, 'logs')
    json_path = os.path.join(log_dir, f'sabr_opt_params_{target_tenor}y.json')
    
    os.makedirs(log_dir, exist_ok=True)
    today_str = datetime.datetime.now().strftime("%Y-%m-%d")
    
    # look for existing sabr params
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            saved_date = data.get('date')
            saved_params = data.get('params')
            print(f"\n[INFO] Found valid SABR params from {saved_date}. Loading...")
            return saved_params
        except Exception as e:
            print(f"[ERROR] Could not read cached params: {e}. Recalibrating.")

    print(f"\n>>> Running SABR Calibration for Tenor {target_tenor}...")
    calibrated_params = calib_func(vol_file, target_tenor=target_tenor)
    
    # JSON serialization helper
    if isinstance(calibrated_params, np.ndarray):
        params_serializable = calibrated_params.tolist()
    elif isinstance(calibrated_params, list):
        params_serializable = [list(p) if isinstance(p, (tuple, np.ndarray)) else p for p in calibrated_params]
    elif isinstance(calibrated_params, dict):
        params_serializable = {str(k): list(v) if isinstance(v, (tuple, np.ndarray)) else v for k, v in calibrated_params.items()}
    else:
        params_serializable = calibrated_params

    save_data = {
        "date": today_str,
        "params": params_serializable
    }
    
    try:
        with open(json_path, 'w') as f:
            json.dump(save_data, f, indent=4)
        print(f"[INFO] New parameters saved to: {json_path}")
    except Exception as e:
        print(f"[WARN] Could not save parameters to JSON: {e}")
        
    return calibrated_params

def parse_tenor(x):
    s = str(x).upper().strip()
    try:
        if s.endswith('Y'): return float(s[:-1])
        if s.endswith('M'): return float(s[:-1])/12.0
        if s == '1Y': return 1.0
    except:
        return 0.0
    return 0.0

def parse_strike(x):
    s = str(x).upper().strip()
    if s == 'ATM': return 0.0
    if s.endswith('BP'):
        try: return float(s[:-2]) / 10000.0
        except: return 0.0
    return 0.0

def _run_pca(C_base, k):
    k = int(k)
    evals, evecs = linalg.eigh(C_base)
    
    # Sort descending
    idx = np.argsort(evals)[::-1]
    evals, evecs = evals[idx], evecs[:, idx]
    
    # # check varaiance
    total_var = np.sum(evals)
    explained_var = np.sum(evals[:k])
    ratio = (explained_var / total_var) * 100
    print(f"[PCA DEBUG] Factors: {k} | Variance Explained: {ratio:.2f}%")

    evals_k = np.maximum(evals[:k], 0.0)
    evecs_k = evecs[:, :k]
    
    C_reduced = evecs_k @ np.diag(evals_k) @ evecs_k.T
    d = np.sqrt(np.diag(C_reduced))
    return C_reduced / np.outer(d, d)
