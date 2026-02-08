import pandas as pd
import numpy as np
from src.plotting import get_vol_surface_plot
from src.calibration import sabr_normal_vol
import time
import os
from src.config import DATA_DIR, LMM_TENOR, RHO_OIS_EUR, CHECK_CALIBRATION, STRIKE, EXERCISE_DATES, \
    DECAY_B, CORR_MODE, RHO_INF, CALIB_TENOR, SIM_STEPS, SIM_PATHS, RFR
from src.utils import load_curve, get_fwd_rates, get_sabr_params
from src.calibration import calibrate_sabr
from src.model import MulticurveSABR_LMM
from src.pricers import bermudan_swaption_pricer, get_greeks
import torch
from src.torch_model import TorchSABR_LMM 
from src.pricers import torch_bermudan_pricer

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
    if RFR:
        eur_curve = ois_curve
    else:
        eur_curve = load_curve(eur_file)
    
    # 3. forward rates
    grid_T, fwd_ois = get_fwd_rates(ois_curve, grid_tau=LMM_TENOR)
    # same grid
    _, fwd_eur = get_fwd_rates(eur_curve, grid_tau=LMM_TENOR)
    print(f"Data Loading Runtime: {time.time() - start_total:.1f} sec")

    start_cali = time.time()
    # 4. SABR calibration 
    calibrated_params = get_sabr_params(vol_file, CALIB_TENOR, calibrate_sabr)
    print(f"Calibration Runtime: {time.time() - start_cali:.1f} sec")

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
        bermudan_swaption_pricer, 
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
    # PYTORCH AAD PRICING
    # ==========================================
    print("\n" + "="*80)
    print(f"{'PRICING':^80}")
    print("="*80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on device: {device.upper()}")

    start_aad = time.time()

    # initialize AAD model
    aad_model = TorchSABR_LMM(
        tenors=grid_T,
        F_ois=fwd_ois,
        F_eur=fwd_eur,
        sabr_params=sabr_list,
        rate_corr_matrix=rate_corr,
        rho_cross=RHO_OIS_EUR,
        device=device
    )

    # specs
    aad_specs = bermudan_specs.copy()

    # price
    aad_price = torch_bermudan_pricer(
        model=aad_model, 
        trade_specs=aad_specs, 
        M_steps=SIM_STEPS, 
        n_paths=SIM_PATHS
    )

    # greeks -> backward pass
    aad_price.backward()    
    end_aad = time.time()
   
    # OIS Delta
    ois_delta_bucketed = aad_model.F0_ois.grad.cpu().numpy()
    total_ois_delta = np.sum(ois_delta_bucketed)

    # EUR Delta
    eur_delta_bucketed = aad_model.F0_eur.grad.cpu().numpy()
    total_eur_delta = np.sum(eur_delta_bucketed)

    # Vega
    vega_bucketed = aad_model.alphas.grad.cpu().numpy()
    total_vega = np.sum(vega_bucketed)

    print("-" * 60)
    print(f"{'Metric':<20} | {'AAD':>12} | {'Standard':>15}")
    print("-" * 60)

    ois_pv01 = total_ois_delta * 0.0001
    eur_pv01 = total_eur_delta * 0.0001
    vega_1bp = total_vega * 0.0001
    
    print(f"{'Price (bps)':<20} | {aad_price*10000:>12.2f} | {res_berm['Base']*10000:>15.2f}")
    print(f"{'OIS Delta (PV01)':<20} | {ois_pv01*10000:>12.2f} | {res_berm['OIS_Delta']*10000:>15.2f}")
    print(f"{'EUR Delta (PV01)':<20} | {eur_pv01*10000:>12.2f} | {res_berm['EUR_Delta']*10000:>15.2f}")
    print(f"{'Vega':<20} | {vega_1bp*10000:>12.2f} | {res_berm['Vega']*10000:>15.2f}")
  
    print("-" * 60)
    print(f"AAD Runtime: {end_aad - start_aad:.4f} sec")
    print("="*60)