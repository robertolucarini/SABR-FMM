import numpy as np
import scipy.linalg as linalg
from src.config import RHO_OIS_EUR, SEED, B_BRIDGE, PREDICT_CORRECT, DECAY_B, \
    ANTITHETIC_VAR, RNG_TYPE, SIM_PATHS, SIM_STEPS, PCA_FACTORS, CORR_MODE
from scipy.stats import qmc, norm
from src.utils import _run_pca

class MulticurveSABR_LMM:
    def __init__(self, tenors, F_ois, F_eur, sabr_params, rate_corr_matrix, 
                 rho_cross=RHO_OIS_EUR, rate_corr_mode=CORR_MODE, n_factors=PCA_FACTORS):
        # way faster than 64
        self.dtype = np.float32
        # times
        self.T = np.array(tenors, dtype=self.dtype)
        # year fraction
        self.tau = np.diff(self.T).astype(self.dtype)
        # number of tenors
        self.N = len(F_ois)

        # market ois (yields)        
        self.F0_ois = np.array(F_ois, dtype=self.dtype)
        # market eur (yields)
        self.F0_eur = np.array(F_eur, dtype=self.dtype)
        
        # sabr params
        # vol
        self.alphas = np.array([p[0] for p in sabr_params], dtype=self.dtype)
        # skew (rate v vol)
        self.rhos = np.array([p[1] for p in sabr_params], dtype=self.dtype)
        # vol of vol
        self.nus = np.array([p[2] for p in sabr_params], dtype=self.dtype)
        
        # rate corr (ois 1M v 3M)
        # NxN
        if rate_corr_mode=="pca":
            C_base_final = self._run_pca(rate_corr_matrix, n_factors)
        else:
            C_base_final = rate_corr_matrix

        # 4Nx4N Cholesky decomposition matrix 
        self.L = self._build_correlation(C_base_final, rho_cross).astype(self.dtype)

        # drift has same corr matrix as rates
        # take only upper triangle -> future tenors
        self.drift_corr_matrix = np.triu(C_base_final, k=1).astype(self.dtype)

    def get_ois_bonds(self):
        """ Calculate ois disc factors"""
        # discount factor for each individual step
        step_dfs = 1.0 / (1.0 + self.tau * self.F0_ois)
        # cumulate
        cum_dfs = np.cumprod(step_dfs)
        # concatenate 1 at the beginning
        return np.r_[1.0, cum_dfs].astype(self.dtype)

    def _build_correlation(self, C_base, rho_cross):
        # N tenors and 4 variables: OIS, EUR, OIS vol, EUR vol
        # tenors
        N = self.N
        # full corr matrix 4Nx4N
        BigC = np.eye(4 * N)
        
        # 1. rate-rate 
        # assumption -> OIS and EUR move with same vol C_base, linked with corr rho_cross
        # OIS 
        BigC[0:N, 0:N] = C_base 
        # EUR 
        BigC[N:2*N, N:2*N] = C_base
        # corr(OIS, EUR) 
        BigC[0:N, N:2*N] = C_base * rho_cross 
        # corr(EUR,OIS) 
        BigC[N:2*N, 0:N] = BigC[0:N, N:2*N].T 
        
        # 2. vol-vol -> same as rate
        # assumption -> corr of vols equal corr or rates         
        # corr(OIS vol,OIS vol)
        BigC[2*N:3*N, 2*N:3*N] = C_base
        # corr(EUR vol,EUR vol)
        BigC[3*N:4*N, 3*N:4*N] = C_base
        # corr(OIS vol,EUR vol)
        BigC[2*N:3*N, 3*N:4*N] = C_base * rho_cross
        # corr(EUR vol,OIS vol)
        BigC[3*N:4*N, 2*N:3*N] = BigC[2*N:3*N, 3*N:4*N].T
        
        # 3. Skew -> links rate with their specific vol (rho_k from sabr)
        for k in range(N):
            # Corr(OIS, OIS vol)
            BigC[k, 2*N+k] = self.rhos[k]
            # Corr(OIS vol, OIS)
            BigC[2*N+k, k] = self.rhos[k]
            # Corr(EUR, EUR vol)
            BigC[N+k, 3*N+k] = self.rhos[k]
            # Corr(EUR vol, EUR) 
            BigC[3*N+k, N+k] = self.rhos[k]
            
            # 4. cross-Skew 
            # chain rule-> corr(OIS,EUR vol)=corr(OIS,EUR)*corr(OIS,OIS vol)  
            val = rho_cross * self.rhos[k]
            # Corr(OIS, EUR vol)
            BigC[k, 3*N+k] = val
            # Corr(EUR vol, OIS)
            BigC[3*N+k, k] = val
            # Corr(EUR, OIS vol)
            BigC[N+k, 2*N+k] = val
            # Corr(OIS VOL, EUR)
            BigC[2*N+k, N+k] = val

        # regularization
        try:
            # BigC could have negative eigenvalues (variance)
            return linalg.cholesky(BigC, lower=True)
        except linalg.LinAlgError:
            # eigenvalue smoothing
            # eigenvalues, eigenvectors of BigC
            evals, evecs = np.linalg.eigh(BigC)
            # fix negative eigenvalues
            evals = np.maximum(evals, 1e-9)
            # fix corr matrix with new eigenvalues
            BigC_fixed = evecs @ np.diag(evals) @ evecs.T
            # previous step break diagonal (no more 1s) 
            # -> it becomes a covariance matrix, not a correlation matrix   
            
            # scaling matrix
            inv_sd = np.diag(1.0 / np.sqrt(np.diag(BigC_fixed)))
            # left scales rows, right scales cols
            BigC_corr = inv_sd @ BigC_fixed @ inv_sd

            # lower triangular cholesky matrix 
            return linalg.cholesky(BigC_corr, lower=True)

    def _compute_drifts_slice(self, F_ois, V_ois, V_eur, slice_idx):
        """ 
        compute drifts only for the active slice of tenors 
        Mercurio 2009 with j>k and standard brownian motion (Normal)
        """

        # only taus from starting cutoff date (slice_idx)
        taus_slice = self.tau[slice_idx:]
        
        # normal (no F_ois in numerator)
        # Normal case (no F_ois in numerator)
        gamma_ois = (taus_slice * V_ois) / (1.0 + taus_slice * F_ois) 
        
        # extract the drift corr only for the remaining indexes
        corr_slice = self.drift_corr_matrix[slice_idx:, slice_idx:]
        
        sum_term = gamma_ois @ corr_slice.T

        # sigma0 * [-sum(ois_corr*tau*sigma1)]
        drift_ois = -V_ois * sum_term
        # assumption -> corr(EUR t+1, EUR t) = rho_cross * corr(OIS t+1, OIS t) 
        drift_eur = -V_eur * RHO_OIS_EUR * sum_term

        return drift_ois, drift_eur

    def update_params(self, F_ois=None, F_eur=None, sabr_params=None):
        """update market parameters without rebuilding the corr matrix"""
        if F_ois is not None:
            self.F0_ois = np.array(F_ois, dtype=self.dtype)
        if F_eur is not None:
            self.F0_eur = np.array(F_eur, dtype=self.dtype)
        if sabr_params is not None:
            self.alphas = np.array([p[0] for p in sabr_params], dtype=self.dtype)

    # ---------------------------
    # Simulation
    # ---------------------------
    def get_sobol_paths(self, total_factors, n_paths=SIM_PATHS, n_steps=SIM_STEPS, seed=SEED):
        """ sobol Quasi Monte Carlo adapted for LMM dimensions """
        # total dimension = time_steps * factors
        dim = n_steps * total_factors

        # sobol generator
        sampler = qmc.Sobol(d=dim, scramble=True, seed=seed)

        # draw from Uniform dist
        u_vec = sampler.random(n=n_paths)
        # from Uniform to Std Normal dist 
        z_vec = norm.ppf(u_vec)

        # reshape from (1xdim) to (n_paths, n_steps, factors)
        z_reshaped = z_vec.reshape(n_paths, n_steps, total_factors)
        
        return z_reshaped.astype(self.dtype)

    def get_brownian_bridge(self, z_input, sim_times):
        # z_input shape: (n_steps, n_paths) 
        # -> treating one factor at a time
        
        n_steps, n_paths = z_input.shape
        W = np.zeros((n_steps + 1, n_paths), dtype=self.dtype)

        # breadth-first search BFS
        # fix end point
        bridge_indices = [n_steps]
        # it maps the indexes used to solve current mid index
        map_dependency = {n_steps: (0, 0)}

        # start with full interval
        queue = [(0, n_steps)]
        # organize the time steps hierachically -> end, mid, quarters, eights
        while queue:
            # extrat first point -> get mid points before any qaurter point (BFS)
            left, right = queue.pop(0)
            # average
            mid = (left + right) // 2

            if mid != left and mid != right:
                # add new mid time step
                bridge_indices.append(mid)
                # it maps the indexes used to solve current mid index
                map_dependency[mid] = (left, right)
                # add sub-interval to the back of the queue 
                queue.append((left, mid))
                queue.append((mid, right))
                
        for i, target_idx in enumerate(bridge_indices):
            # best quality sobol (bridge indices) mapped to first time step (target_idex)            
            # initial shocks
            z_val = z_input[i, :]
            
            if i == 0:
                # full period
                T = sim_times[n_steps]
                # W_T = sqrt(T) * Z
                W[target_idx, :] = np.sqrt(T) * z_val
            else:
                # indexes (start/end, quartes, etc)
                left, right = map_dependency[target_idx]
                # current, left, right time values
                ti, tj, tk = sim_times[left], sim_times[target_idx], sim_times[right]
                # delta time steps -> periods
                dt_ki, dt_ji, dt_kj = tk - ti, tj - ti, tk - tj
                # epxected value -> average bwt left and right
                mu = (dt_kj / dt_ki) * W[left, :] + (dt_ji / dt_ki) * W[right, :]
                # conditional var
                sigma = np.sqrt((dt_ji * dt_kj) / dt_ki)
                # intermediate point -> expectation + noise
                W[target_idx, :] = mu + sigma * z_val

        # Return increments (dW)
        return np.diff(W, axis=0)

    def prepare_drivers(self, M_steps=SIM_STEPS, n_paths=SIM_PATHS, seed=SEED,rng_type=RNG_TYPE, antithetic=True, use_bb=B_BRIDGE):
        """ generates correlated brownian increments dW"""

        np.random.seed(seed)
        dt = self.T[-1] / M_steps
        sq_dt = np.sqrt(dt, dtype=self.dtype)
        time_grid = np.linspace(0, self.T[-1], M_steps + 1, dtype=self.dtype)
        
        # anthitetic variates needs half paths 
        if antithetic: 
            n_gen = n_paths // 2 
        else: 
            n_paths

        # number of stoch processes
        total_factors = 4 * self.N

        # get uncorrelated dW        
        if rng_type == 'sobol':
            Z_gen = self.get_sobol_paths(total_factors, n_paths=n_gen)
        else:
            # standard
            Z_gen = np.random.normal(size=(n_gen, M_steps, total_factors)).astype(self.dtype)
            
        # antithetic variates logic
        if antithetic:
            Z = np.concatenate([Z_gen, -Z_gen], axis=0)
            # handle odd n_paths case just in case
            if Z.shape[0] > n_paths: 
                Z = Z[:n_paths]
        else:
            Z = Z_gen

        # correlated shocks preparation
        if use_bb:
            Z_bridged = np.zeros_like(Z)
            for i in range(total_factors):
                # build z for a single process
                z_factor = Z[:, :, i].T
                # dw already scaled by sqrt(dt)  
                dw_factor = self.get_brownian_bridge(z_factor, time_grid)
                Z_bridged[:, :, i] = dw_factor.T
            # correlate with cholesky L
            dW = Z_bridged @ self.L.T
        else:
            dW = (Z @ self.L.T) * sq_dt
        
        # return stoch increments    
        return dW

    def simulate(self, M_steps=SIM_STEPS, n_paths=SIM_PATHS, seed=SEED,rng_type=RNG_TYPE, antithetic=ANTITHETIC_VAR, use_bb=B_BRIDGE, dW_custom=None):
        """ Monte Carlo engine"""

        np.random.seed(seed)
        dt = self.dtype(self.T[-1] / M_steps)
        time_grid = np.linspace(0, self.T[-1], M_steps + 1, dtype=self.dtype)
        
        # tensor[paths, time_steps, tenors]
        F_ois = np.zeros((n_paths, M_steps+1, self.N), dtype=self.dtype)
        F_eur = np.zeros((n_paths, M_steps+1, self.N), dtype=self.dtype)
        V_ois = np.zeros((n_paths, M_steps+1, self.N), dtype=self.dtype)
        V_eur = np.zeros((n_paths, M_steps+1, self.N), dtype=self.dtype)
        
        # initalize at current values
        F_ois[:, 0, :] = self.F0_ois
        F_eur[:, 0, :] = self.F0_eur
        V_ois[:, 0, :] = self.alphas
        # assumption -> deterministic constant basis spread
        V_eur[:, 0, :] = self.alphas

        # brownian increments 
        # reuse pre-calculated drivers (for greeks)
        if dW_custom is not None:
            dW = dW_custom
        else:
            dW = self.prepare_drivers(M_steps, n_paths, seed, rng_type, antithetic, use_bb)
        
        # precalculate constant term for vol optimization
        vol_drift_correction = (-0.5 * self.nus**2 * dt).astype(self.dtype)
        
        for t in range(M_steps):
            t_curr = time_grid[t]
            
            # copy previous state to current state
            F_ois[:, t+1] = F_ois[:, t]
            F_eur[:, t+1] = F_eur[:, t]
            V_ois[:, t+1] = V_ois[:, t]
            V_eur[:, t+1] = V_eur[:, t]

            # active set strategy -> find the first tenor that is still alive (fixing time > t_curr)
            start_idx = np.searchsorted(self.T[:self.N], t_curr, side='right')
            
            if start_idx >= self.N:
                continue # all tenors expired

            # shape becomes (n_paths, N_alive)
            fo = F_ois[:, t, start_idx:]
            fe = F_eur[:, t, start_idx:]
            vo = V_ois[:, t, start_idx:]
            ve = V_eur[:, t, start_idx:]
            
            # slice parameters
            curr_nus = self.nus[start_idx:]
            
            # slice increments
            # global column map: 0..N (fo), N..2N (fe), 2N..3N (vo), 3N..4N (ve)
            dW_fo = dW[:, t, 0 + start_idx : self.N]
            dW_fe = dW[:, t, self.N + start_idx : 2*self.N]
            dW_vo = dW[:, t, 2*self.N + start_idx : 3*self.N]
            dW_ve = dW[:, t, 3*self.N + start_idx : 4*self.N]
            
            # control for very low vol
            vo = np.maximum(vo, 1e-6) 
            ve = np.maximum(ve, 1e-6)
            
            curr_drift_correction = vol_drift_correction[start_idx:]
            
            # vol evolution
            vo_next = vo * np.exp(curr_drift_correction + curr_nus*dW_vo)
            ve_next = ve * np.exp(curr_drift_correction + curr_nus*dW_ve)
            
            # drifts-> calls the sliced optimized helper
            mu_ois, mu_eur = self._compute_drifts_slice(fo, vo, ve, start_idx)

            if PREDICT_CORRECT:
                # 1. Predictor step (Euler)
                fo_pred = fo + mu_ois*dt + vo*dW_fo
                # 2. Re-compute drifts at predictor 
                mu_ois_pred, mu_eur_pred = self._compute_drifts_slice(fo_pred, vo, ve, start_idx)
                # 3. Average drifts
                mu_ois = 0.5 * (mu_ois + mu_ois_pred)
                mu_eur = 0.5 * (mu_eur + mu_eur_pred)
            
            # rate evolution
            fo_next = fo + mu_ois*dt + vo*dW_fo
            fe_next = fe + mu_eur*dt + ve*dW_fe

            # update ONLY active slices in the next time step
            F_ois[:, t+1, start_idx:] = fo_next
            F_eur[:, t+1, start_idx:] = fe_next
            V_ois[:, t+1, start_idx:] = vo_next
            V_eur[:, t+1, start_idx:] = ve_next
            
        return time_grid, F_ois, F_eur
