import numpy as np
from src.config import SIM_STEPS, SIM_PATHS, RHO_OIS_EUR, SEED, RNG_TYPE, ANTITHETIC_VAR, B_BRIDGE, BUMP_DELTA, BUMP_VOL
import torch
import torch.nn as nn


def bermudan_swaption_pricer(model, paths_ois, paths_eur, time_grid, trade_specs):
    # unpack specs
    strike = trade_specs['Strike']
    dates = trade_specs['Ex_Dates']
    # Control Variate input
    euro_analytical = trade_specs.get('Euro_Analytical', None) 

    n_paths = paths_ois.shape[0]
    ex_steps = [np.abs(time_grid - t).argmin() for t in dates]
    cashflows = np.zeros(n_paths)
    
    # store european payoff for Control Variate
    euro_vals = np.zeros(n_paths)

    # backwards induction (LSM)
    for i in range(len(ex_steps)-1, -1, -1):
        step = ex_steps[i]
        k_start = np.abs(model.T[:-1] - time_grid[step]).argmin()
        
        # forward (simulated) ois
        f_ois_t = paths_ois[:, step, :]
        taus = model.tau[k_start:]
        # discounting to current step
        dfs_ois = np.cumprod(1.0 / (1 + taus * f_ois_t[:, k_start:]), axis=1)
        
        # forward (simulated) eur 
        f_eur_t = paths_eur[:,  step, k_start:]
        # payoff (payer swap)
        payoff = (f_eur_t - strike) * taus
        # intrinsic value at step i (exercise now)
        intrinsic = np.maximum(np.sum(dfs_ois * payoff, axis=1), 0)
        
        # european value at last exercise date (for control variate)
        if i == len(ex_steps)-1:
            # deflate european payoff -> martingale under terminal measure 
            euro_vals = intrinsic / dfs_ois[:, -1]

        # deflate by P(t, TN) -> martingale under terminal measure 
        intrinsic_neutral = intrinsic / dfs_ois[:, -1]
        
        if i == len(ex_steps)-1:
            # at maturity, continuation = 0  
            continuation = np.zeros(n_paths)
        else:
            # regression
            annuity = np.sum(dfs_ois * taus, axis=1)
            par_rate = np.sum(dfs_ois * f_eur_t * taus, axis=1) / annuity
            
            # basis functions -> X = [1, S, S^2, S*A]
            X = np.column_stack([np.ones(n_paths), par_rate, par_rate**2, par_rate * annuity])
            
            # in-the-money mask
            itm = intrinsic_neutral > 0
            # initiate continuation value
            continuation = np.zeros(n_paths)
            
            if np.sum(itm) > 50:
                # regression only itm paths
                X_itm = X[itm]
                y_itm = cashflows[itm]
                try:
                    # only coeffs needed
                    coeffs, _, _, _ = np.linalg.lstsq(X_itm, y_itm, rcond=None)
                    # update continuation value only for itm paths 
                    continuation[itm] = X_itm @ coeffs
                except:
                    pass

        # exercise decision
        do_ex = intrinsic_neutral > continuation
        # upadte only exercising paths
        cashflows[do_ex] = intrinsic_neutral[do_ex]

    # discount factor to today
    P_0_TN = model.get_ois_bonds()[-1]
    # bermudan price today
    berm_mc_price = P_0_TN * np.mean(cashflows)

    # Control Variate
    if euro_analytical is not None:
        # MC european price
        euro_mc_price = P_0_TN * np.mean(euro_vals)
        
        # Beta = Cov(Berm, Euro) / Var(Euro)
        beta = 1.0 
        
        # adjust price
        cv_adj = euro_mc_price - euro_analytical

        # new price = MC_Berm - Beta * (MC_Euro - Analytical_Euro)
        adjusted_price = berm_mc_price - beta * cv_adj

        print(f"Bermudan (Adj) price: {berm_mc_price:.4f} | MC European : {euro_mc_price:.4f} | Analytical European: {euro_analytical:.4f} | Control Variate Adj: {cv_adj:.4f}")
        return adjusted_price
    
    return berm_mc_price

def get_greeks(pricer_func, trade_specs, model_cls, grid_T, fwd_ois, fwd_eur,sabr_p, rate_corr, cross_rho=RHO_OIS_EUR):
          
    # instantiate model
    model = model_cls(grid_T, fwd_ois, fwd_eur, sabr_p, rate_corr, cross_rho)
    
    # precalculate common random numbers 
    dW_frozen = model.prepare_drivers(
        M_steps=SIM_STEPS, 
        n_paths=SIM_PATHS, 
        seed=SEED, 
        rng_type=RNG_TYPE, 
        antithetic=ANTITHETIC_VAR, 
        use_bb=B_BRIDGE)
    
    # simulation helper
    def run_sim_and_price():
        t, p_o, p_e = model.simulate(
            M_steps=SIM_STEPS, 
            n_paths=SIM_PATHS, 
            dW_custom=dW_frozen)
        return pricer_func(model, p_o, p_e, t, trade_specs)

    # Base Price
    base_price = run_sim_and_price()
    
    res = {'Base': base_price, 'OIS_Delta': 0, 'EUR_Delta': 0, 'Vega': 0}

    # OIS Delta
    model.update_params(F_ois = fwd_ois + BUMP_DELTA)
    res['OIS_Delta'] = run_sim_and_price() - base_price
    # revert to original params
    model.update_params(F_ois = fwd_ois)

    # EUR Delta
    model.update_params(F_eur = fwd_eur + BUMP_DELTA)
    res['EUR_Delta'] = run_sim_and_price() - base_price
    model.update_params(F_eur = fwd_eur)
        
    # Vega
    bump_sabr = [(p[0] + BUMP_VOL, p[1], p[2]) for p in sabr_p]
    model.update_params(sabr_params = bump_sabr)
    res['Vega'] = run_sim_and_price() - base_price
    model.update_params(sabr_params = sabr_p)
    
    return res

def torch_bermudan_pricer(model, trade_specs, M_steps=SIM_STEPS, n_paths=SIM_PATHS):
    # simulate rates
    time_grid, paths_ois, paths_eur = model.simulate(M_steps, n_paths)
    
    strike = trade_specs['Strike']
    ex_dates = torch.tensor(trade_specs['Ex_Dates'], device=model.device)
    
    # map dates to steps
    ex_steps = []
    for t in ex_dates:
        dist = torch.abs(time_grid - t)
        ex_steps.append(torch.argmin(dist))
        
    n_paths = paths_ois.shape[0]
    
    # cashflows -> V(t) / P(t, T_N)
    deflated_cashflows = torch.zeros(n_paths, device=model.device, dtype=torch.float64)
    
    # backwards induction
    for i in range(len(ex_steps) - 1, -1, -1):
        step = ex_steps[i]
        
        # tenors
        t_val = time_grid[step]
        k_start = int(torch.sum(model.T[:-1] <= t_val).item())
        
        # rates
        f_ois_t = paths_ois[:, step, :]
        f_eur_t = paths_eur[:, step, :]
        # dt
        taus = model.tau[k_start:]
        
        # Numeraire Ratio: P(t, T_N) / P(t, T_start) 
        # -> discount factor from T_N back to current time t
        curve_dfs = 1.0 / (1.0 + taus * f_ois_t[:, k_start:])
        
        # (Paths, Tenors) -> discount to T_N (last tenor)
        p_t_Tn = torch.prod(curve_dfs, dim=1)
        
        # cumulative disc fator
        swap_dfs = torch.cumprod(curve_dfs, dim=1)
        # swap payoff
        swap_val = torch.sum(swap_dfs * (f_eur_t[:, k_start:] - strike) * taus, dim=1)
        
        # option intrinsic value
        intrinsic = torch.maximum(swap_val, torch.tensor(0.0, device=model.device))
        
        # deflate to Terminal Measure -> P(t, T_N)
        intrinsic_deflated = intrinsic / p_t_Tn
        
        # regression
        if i == len(ex_steps) - 1:
            # last period continuation value is zer0
            continuation_deflated = torch.zeros(n_paths, device=model.device)
        else:
            annuity = torch.sum(swap_dfs * taus, dim=1)
            par_rate = torch.sum(swap_dfs * f_eur_t[:, k_start:] * taus, dim=1) / annuity
            # state variables
            X0 = torch.ones(n_paths, device=model.device)
            X1 = par_rate
            X2 = par_rate**2
            X3 = par_rate * annuity
            # basis function
            X = torch.stack([X0, X1, X2, X3], dim=1)
            
            # in-the-money mask
            itm_mask = intrinsic_deflated > 0
            
            if itm_mask.sum() > 50:
                X_itm = X[itm_mask].detach() 
                Y_itm = deflated_cashflows[itm_mask].detach()
                # least-squares                
                sol = torch.linalg.lstsq(X_itm, Y_itm).solution
                continuation_deflated = torch.zeros(n_paths, device=model.device)
                # continuation value as basis function * reg coeffs
                continuation_deflated[itm_mask] = X[itm_mask] @ sol
            else:
                continuation_deflated = torch.zeros(n_paths, device=model.device)

        # decision mask
        do_ex = intrinsic_deflated > continuation_deflated
        
        # update -> if exercise, take intrinsic else keep previous
        deflated_cashflows = torch.where(do_ex, intrinsic_deflated, deflated_cashflows)

    # disc factor from maturity
    p_0_Tn = model.get_terminal_bond()
    
    # discount using terminal measure
    return p_0_Tn * torch.mean(deflated_cashflows)