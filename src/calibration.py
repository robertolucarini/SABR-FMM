import numpy as np
import polars as pl
from scipy.optimize import minimize
from src.utils import robust_read_csv, parse_strike, parse_tenor

def sabr_normal_vol(strike_offset, T_ex, alpha, rho, volvol):
    """ implied Normal volatility, Hagan expansion"""

    # alpha: initial instantaneuos vol
    # rho: corr between rate and vol (skew)
    # nu: vol of vol (smile)
    # -> assumption: Beta = 0

    # strike data are K-F
    f_minus_k = -strike_offset
    if abs(f_minus_k) < 1e-8:
        term2 = (2 - 3 * rho**2) / 24 * volvol**2 * T_ex
        return alpha * (1 + term2)

    zeta = (volvol / alpha) * f_minus_k
    # handle atm case
    if abs(zeta) < 1e-8:
        # L'Hopital: if zeta->0 => zeta/x_zeta=1
        term1 = alpha 
        term2 = (2 - 3 * rho**2) / 24 * volvol**2 * T_ex
        return term1 * (1 + term2)

    discriminant = 1 - 2*rho*zeta + zeta**2
    if discriminant <= 0: 
        return alpha 
    
    x_zeta = np.log((np.sqrt(discriminant) + zeta - rho) / (1 - rho))
    term1 = alpha * (zeta / x_zeta)
    term2 = (2 - 3 * rho**2) / 24 * volvol**2 * T_ex
    return term1 * (1 + term2)

def calibrate_sabr(vol_csv_path, target_tenor=1):
    print(f"Loading Vols: {vol_csv_path}")
    df = robust_read_csv(vol_csv_path)

    c_exp = next((c for c in df.columns if 'EXPIRY' in c), None)
    c_ten = next((c for c in df.columns if 'UNDERLYINGTENOR' in c or 'TENOR' in c), None)
    c_str = next((c for c in df.columns if 'STRIKE' in c), None)
    c_val = next((c for c in df.columns if ('VALUE' in c or 'VOL' in c) and 'ID' not in c), None)

    if not all([c_exp, c_ten, c_str, c_val]): 
        raise ValueError("Missing Vol Columns.")

    df = df.with_columns([
        pl.col(c_exp).map_elements(parse_tenor, return_dtype=pl.Float64).alias('T_Exp'),
        pl.col(c_ten).map_elements(parse_tenor, return_dtype=pl.Float64).alias('T_Ten'),
        pl.col(c_str).map_elements(parse_strike, return_dtype=pl.Float64).alias('Offset'),
        pl.col(c_val).cast(pl.Float64, strict=False).alias('Value')
    ])
    
    df = df.drop_nulls(subset=['Value', 'T_Exp', 'T_Ten'])
    
    # filter target tenor
    subset = df.filter((pl.col('T_Ten') - target_tenor).abs() < 0.05)
    
    if subset.is_empty(): 
        print(f"Warning: Tenor {target_tenor}Y not found. Defaulting to 1Y slice.")
        subset = df.filter((pl.col('T_Ten') - 1.0).abs() < 0.05)

    params_dict = {}

    print("\n" + "="*80)
    print(f"{'SABR CALIBRATION REPORT':^80}")
    print(f"{f'Target Tenor: {target_tenor}Y':^80}")
    print("="*80)
    print(f"{'Expiry':<8} | {'Alpha':<10} | {'Rho':<10} | {'Nu (VolVol)':<12} | {'RMSE (bps)':<12} | {'Status':<10}")
    print("-" * 80)

    # unique expiries
    unique_expiries = subset['T_Exp'].unique().sort().to_list()

    for T in unique_expiries:
        slice_df = subset.filter(pl.col('T_Exp') == T)
        offsets = slice_df['Offset'].to_numpy()
        mkt_vols = slice_df['Value'].to_numpy()
        
        def obj(p):
            alpha_, rho_, volvol_ = p
            # penalties
            if alpha_ <= 0 or abs(rho_) >= 0.999 or abs(volvol_) >= 20.0: 
                return 1e9
            # get model implied vol using hagan formula
            model_vols = np.array([sabr_normal_vol(k, T, alpha_, rho_, volvol_) for k in offsets])
            # sum of squared errors SSE
            return np.sum((model_vols - mkt_vols)**2)

        # initial sabr params    
        start_alpha = np.mean(mkt_vols)
        initial_guess = [start_alpha, 0.0, 0.1]
        
        try:
            res = minimize(obj, initial_guess, method='Nelder-Mead', tol=1e-6)
            alpha, rho, nu = res.x
            params_dict[T] = tuple(res.x)

            final_sse = res.fun
            if final_sse >= 1e8:
                sse_bps = 999.99
                status = "FAILED"
            else:
                rmse = np.sqrt(final_sse / len(offsets))
                rmse_bps = rmse * 10000 
                status = "OK" if res.success else "WARN"
        except Exception as e:
            alpha, rho, nu = 0, 0, 0
            rmse_bps = 999.99
            status = "ERR"

        print(f"{T:<8.2f} | {alpha:<10.6f} | {rho:<10.4f} | {nu:<12.4f} | {rmse_bps:<12.4f} | {status:<10}")

    print("="*80 + "\n")
        
    return params_dict