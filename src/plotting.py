import polars as pl
import pandas as pd 
import numpy as np
from src.calibration import sabr_normal_vol
from src.utils import robust_read_csv

def parse_tenor(s):
    s = str(s).upper().strip()
    try:
        if s.endswith('Y'): return float(s[:-1])
        if s.endswith('M'): return float(s[:-1])/12.0
        if s == '1Y': return 1.0
    except:
        pass
    return 0.0

def get_vol_surface_plot(vol_file, calibrated_params, tenor_step):
    import matplotlib.pyplot as plt
    from matplotlib import cm
    import seaborn as sns

    print(f"Running diagnostics for {vol_file}...")
    
    df = robust_read_csv(vol_file)
    
    c_exp = next((c for c in df.columns if 'EXPIRY' in c), None)
    c_ten = next((c for c in df.columns if 'UNDERLYING' in c or 'TENOR' in c), None)
    c_str = next((c for c in df.columns if 'STRIKE' in c), None)
    c_val = next((c for c in df.columns if 'VOL' in c or 'VALUE' in c), None)
    
    if not all([c_exp, c_ten, c_str, c_val]):
        print("[WARN] Could not find all columns for diagnostics.")
        return

    df = df.with_columns([
        pl.col(c_ten).map_elements(parse_tenor, return_dtype=pl.Float64).alias('Tenor_Val'),
        pl.col(c_exp).map_elements(parse_tenor, return_dtype=pl.Float64).alias('Expiry_Val')
    ])
    
    df_slice = df.filter((pl.col('Tenor_Val') - float(tenor_step)).abs() < 0.05)
    
    if df_slice.height == 0:
        print(f"[WARN] No data found for Tenor {tenor_step}Y.")
        return

    def parse_strike(x):
        s = str(x).upper().strip()
        if s == 'ATM': return 0.0
        if s.endswith('BP'): 
            try: return float(s[:-2])
            except: return 0.0
        return 0.0
        
    df_slice = df_slice.with_columns([
        pl.col(c_str).map_elements(parse_strike, return_dtype=pl.Float64).alias('Strike_Bp'),
        (pl.col(c_val).cast(pl.Float64, strict=False) * 10000).alias('Vol_Bps')
    ])
    
    pivot_df_pl = df_slice.pivot(values='Vol_Bps', index='Expiry_Val', columns='Strike_Bp', aggregate_function='first').sort('Expiry_Val')
    
    pivot_mkt = pivot_df_pl.to_pandas().set_index('Expiry_Val')
    pivot_mkt.columns = pivot_mkt.columns.astype(float)
    pivot_mkt = pivot_mkt.sort_index(axis=1)

    pivot_model = pd.DataFrame(index=pivot_mkt.index, columns=pivot_mkt.columns)
    
    params_map = {}
    if isinstance(calibrated_params, dict):
        for k, v in calibrated_params.items():
            try:
                k_float = float(k) if isinstance(k, (int, float)) else parse_tenor(k)
                params_map[k_float] = v
            except:
                pass

    for t_exp in pivot_mkt.index:
        if not params_map: continue
        avail_ts = np.array(list(params_map.keys()))
        idx = (np.abs(avail_ts - t_exp)).argmin()
        nearest_t = avail_ts[idx]
        
        if abs(nearest_t - t_exp) > 0.5: continue
            
        alpha, rho, nu = params_map[nearest_t]
        
        for strike_bp in pivot_mkt.columns:
            k_offset = strike_bp / 10000.0
            vol_model = sabr_normal_vol(k_offset, t_exp, alpha, rho, nu)
            pivot_model.loc[t_exp, strike_bp] = vol_model * 10000 

    pivot_model = pivot_model.astype(float)
    residuals = pivot_model - pivot_mkt
    
    X_str = pivot_mkt.columns.values
    Y_exp = pivot_mkt.index.values
    X, Y = np.meshgrid(X_str, Y_exp)
    Z_mkt = pivot_mkt.values
    Z_mod = pivot_model.values
    Z_res = residuals.values
    
    max_res = np.nanmax(np.abs(Z_res))
    if np.isnan(max_res) or max_res == 0: max_res = 1.0
    res_lim = max_res * 1.1

    fig1 = plt.figure(figsize=(24, 8))
    
    ax1 = fig1.add_subplot(1, 3, 1, projection='3d')
    surf1 = ax1.plot_surface(X, Y, Z_mkt, cmap=cm.viridis, linewidth=0, antialiased=False, alpha=0.9)
    ax1.set_title(f'market vol surface (bps) | tenor: {tenor_step}y')
    ax1.set_xlabel('strike (bps)')
    ax1.set_ylabel('expiry (y)')
    ax1.set_zlabel('vol (bps)')
    fig1.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10)

    ax2 = fig1.add_subplot(1, 3, 2, projection='3d')
    surf2 = ax2.plot_surface(X, Y, Z_mod, cmap=cm.viridis, linewidth=0, antialiased=False, alpha=0.9)
    ax2.set_title(f'SABR vol surface (bps) | tenor: {tenor_step}y')
    ax2.set_xlabel('strike (bps)')
    ax2.set_ylabel('expiry (Y)')
    ax2.set_zlabel('vol (bps)')
    fig1.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10)
    
    ax3 = fig1.add_subplot(1, 3, 3, projection='3d')
    surf3 = ax3.plot_surface(X, Y, Z_res, cmap=cm.coolwarm, antialiased=False, alpha=0.9)
    ax3.plot_surface(X, Y, np.zeros_like(Z_res), color='black', alpha=0.9)
    ax3.set_title(f"residuals (model - market) | tenor: {tenor_step}y")
    ax3.set_xlabel("strike (bps)")
    ax3.set_ylabel("expiry (y)")
    ax3.set_zlabel("error (bps)")
    ax3.set_zlim(-res_lim, res_lim)
    fig1.colorbar(surf3, ax=ax3, shrink=0.5, aspect=10)

    fig1.tight_layout()
    plt.savefig(f"pics/calibration_surfaces_with_residuals_{tenor_step}y.png")
    plt.close()
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(residuals, annot=True, fmt=".1f", cmap="RdBu_r", center=0)
    plt.title(f"SABR calibration errors (bps) - tenor {tenor_step}y")
    plt.ylabel("expiry (y)")
    plt.xlabel("strike (bps)")
    plt.savefig(f"pics/calibration_heatmap_{tenor_step}y.png")
    plt.close()
    
    print(f"Diagnostics plots saved: pics/calibration_heatmap_{tenor_step}y.png, calibration_surfaces_with_residuals_{tenor_step}y.png")