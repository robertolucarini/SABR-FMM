import pandas as pd
import numpy as np
from src.plotting import get_vol_surface_plot
from src.calibration import sabr_normal_vol
import time
import os
from src.config import DATA_DIR, LMM_TENOR, RHO_OIS_EUR, CHECK_CALIBRATION, STRIKE, EXERCISE_DATES, \
    DECAY_B, CORR_MODE, RHO_INF, CALIB_TENOR
from src.utils import load_curve, get_fwd_rates, get_sabr_params
from src.calibration import calibrate_sabr
from src.model import MulticurveSABR_LMM
from src.pricers import _bermudan_swaption_pricer, get_greeks


if __name__ == "__main__":
    start_total = time.time()
    
    print("\n\n\n" + "="*80)
    print(f"{'DATA':^80}")
    print("="*80)
    
    # 1. paths
    estr_file = os.path.join(DATA_DIR, "estr_disc.csv")
    eur_file  = os.path.join(DATA_DIR, "eur6m_disc.csv")
    vol_file  = os.path.join(DATA_DIR, "estr_vol_full_strikes.csv")
    
    # 2. load curves
    ois_curve = load_curve(estr_file, "OIS ESTR")
    eur_curve = load_curve(eur_file)
    
    # 3. forward rates
    grid_T, fwd_ois = get_fwd_rates(ois_curve, grid_tau=LMM_TENOR)
    # same grid
    _, fwd_eur = get_fwd_rates(eur_curve, grid_tau=LMM_TENOR)
    
    # 4. SABR calibration 
    calibrated_params = get_sabr_params(vol_file, CALIB_TENOR, calibrate_sabr)

    if CHECK_CALIBRATION:
        print("\n" + "="*80)
        print(f"{'CALIBRATION DIAGNOSTICS':^80}")
        print("="*80)
        
        get_vol_surface_plot(
            vol_file=vol_file,
            calibrated_params=calibrated_params,
            tenor_step=LMM_TENOR)

    # 5. assign each fwd rate in the grid to its specific sabr param
    sabr_list = []
    
    # allows for multiple sets of params via dicts
    if isinstance(calibrated_params, dict):
        # extract keys and convert to float
        keys_float = [float(k) for k in calibrated_params.keys()]
        keys_str = list(calibrated_params.keys())
        # sort them together to keep alignment + zip them, sort by float value, unzip
        sorted_pairs = sorted(zip(keys_float, keys_str), key=lambda x: x[0])
        avail_Ts_float = np.array([p[0] for p in sorted_pairs])
        avail_Ts_str   = [p[1] for p in sorted_pairs]
        
        for t in grid_T[:-1]:
            # find nearest tenor in float space
            idx = (np.abs(avail_Ts_float - t)).argmin()
            # retrieve param using the original string key
            key_to_use = avail_Ts_str[idx]
            sabr_list.append(calibrated_params[key_to_use])
    else:
        # fixed params for each tenor for debugging
        sabr_list = [calibrated_params] * len(fwd_ois)

    # 6. build instantaneuos correlation matrix (C_base, block for bigger corr matrix in LMM)
    N = len(fwd_ois)
    # 1d array of evenly spaced indexes
    idx = np.arange(N)
    # create ghost col (None) then subtract Nx2 and 1xN to get the time diff between index pairs 
    dist = np.abs(idx[:, None] - idx)

    if (CORR_MODE == "exp") | (CORR_MODE == "pca"):
        rate_corr = np.exp(-DECAY_B * dist)
    elif CORR_MODE == "two_param":
        # two-parameter formula
        rate_corr = RHO_INF + (1 - RHO_INF) * np.exp(-DECAY_B * dist)

    # 7. initialize model
    # model = MulticurveSABR_LMM(grid_T, fwd_ois, fwd_eur, sabr_list, rate_corr)

    # ==========================================
    # BERMUDAN SWAPTION
    # ==========================================
    bermudan_specs = {
        'Type': 'Bermudan Swaption',
        'Strike': STRIKE, #np.mean(fwd_eur), # ATM Strike
        'Ex_Dates': EXERCISE_DATES
    }
    
    res_berm = get_greeks(
        _bermudan_swaption_pricer, 
        bermudan_specs, 
        MulticurveSABR_LMM, 
        grid_T, 
        fwd_ois, 
        fwd_eur, 
        sabr_list, 
        rate_corr, 
        RHO_OIS_EUR 
    )
    
    # ==========================================
    # FINAL REPORT
    # ==========================================
    print("\n" + "="*80)
    print(f"{'VALUATION REPORT':^80}")
    print("="*80)
    
    def print_row(label, val1, is_money=False):
        fmt = "{:>12.2f}" if is_money else "{:>12.6f}"
        print(f"{label:<20} | {fmt.format(val1)}")

    print(f"{'Metric':<20} | {'Bermudan':>12} ")
    print("-" * 60)
    
    # Scale numbers for display (bps or cents)
    print_row("Bermudan (bps)", res_berm['Base']*10000, True)
    print_row("OIS Delta (PV01)", res_berm['OIS_Delta']*10000, True)
    print_row("EUR Delta (PV01)", res_berm['EUR_Delta']*10000, True)
    print_row("Vega (1bp)", res_berm['Vega']*10000, True)
    
    print("-" * 60)
    print(f"Total Runtime: {time.time() - start_total:.1f} sec")
    print("="*60)