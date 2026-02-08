import torch
import torch.nn as nn
import numpy as np
from src.config import RHO_OIS_EUR, SEED, SIM_PATHS, SIM_STEPS

torch.set_default_dtype(torch.float64)

class TorchSABR_LMM(nn.Module):
    def __init__(self, tenors, F_ois, F_eur, sabr_params, rate_corr_matrix, rho_cross=RHO_OIS_EUR, device='cpu'):
        super().__init__()
        self.device = device
        
        # constants => buffer 
        self.register_buffer('T', torch.tensor(tenors, device=device))
        self.register_buffer('tau', torch.tensor(np.diff(tenors), device=device))
        self.N = len(F_ois)

        # convert inputs to parameters (leaf nodes for AD)
        # ndarrays => tensors
        self.F0_ois = torch.tensor(F_ois, device=device, requires_grad=True)
        self.F0_eur = torch.tensor(F_eur, device=device, requires_grad=True)
        
        # SABR params
        self.alphas = torch.tensor([p[0] for p in sabr_params], device=device, requires_grad=True)
        self.rhos = torch.tensor([p[1] for p in sabr_params], device=device, requires_grad=True)
        self.nus = torch.tensor([p[2] for p in sabr_params], device=device, requires_grad=True)

        # correlation matrix
        self.L = self._build_correlation(rate_corr_matrix, rho_cross)
        
        # drift correlation (upper triangle)
        C_base = torch.tensor(rate_corr_matrix, device=device)
        # constants
        self.register_buffer('drift_corr_matrix', torch.triu(C_base, diagonal=1))
        self.register_buffer('rho_ois_eur', torch.tensor(rho_cross, device=device))

    def _build_correlation(self, C_base_np, rho_cross=RHO_OIS_EUR):
        N = self.N
        C_base = torch.tensor(C_base_np, device=self.device)
        # 1 on diagonal, 0 otherwise
        BigC = torch.eye(4 * N, device=self.device)
        
        # Rate-Rate
        BigC[0:N, 0:N] = C_base
        BigC[N:2*N, N:2*N] = C_base
        BigC[0:N, N:2*N] = C_base * rho_cross
        BigC[N:2*N, 0:N] = BigC[0:N, N:2*N].T
        
        # Vol-Vol
        BigC[2*N:3*N, 2*N:3*N] = C_base
        BigC[3*N:4*N, 3*N:4*N] = C_base
        BigC[2*N:3*N, 3*N:4*N] = C_base * rho_cross
        BigC[3*N:4*N, 2*N:3*N] = BigC[2*N:3*N, 3*N:4*N].T

        # Skew
        rhos = self.rhos.detach()
        for k in range(N):
            BigC[k, 2*N+k] = rhos[k]
            BigC[2*N+k, k] = rhos[k]
            BigC[N+k, 3*N+k] = rhos[k]
            BigC[3*N+k, N+k] = rhos[k]
            
            val = rho_cross * rhos[k]
            BigC[k, 3*N+k] = val
            BigC[3*N+k, k] = val
            BigC[N+k, 2*N+k] = val
            BigC[2*N+k, N+k] = val
            
        # eigenvalue smoothing
        jitter = 1e-7
        try:
            return torch.linalg.cholesky(BigC + jitter * torch.eye(4*N, device=self.device))
        except torch.linalg.LinAlgError:
            # spectral decomposition fix
            print("[WARN] Matrix not Positive Definite. Applying Eigenvalue Smoothing...")
            # eigen decomposition 
            w, v = torch.linalg.eigh(BigC)
            
            # floored eigenvalues
            epsilon = 1e-5
            w_clamped = torch.maximum(w, torch.tensor(epsilon, device=self.device))
            
            # reconstruct matrix: V * diag(w) * V'
            BigC_fixed = v @ torch.diag(w_clamped) @ v.mT
            
            # rescale to ensure diagonal is exactly 1
            # => D = diag(1/sqrt(diag(BigC)))
            d = 1.0 / torch.sqrt(torch.diag(BigC_fixed))
            # PS: can be done by slicing or by diag(diag) as in model.py
            BigC_corr = d[:, None] * BigC_fixed * d[None, :]
            
            # retry Cholesky on fixed matrix
            return torch.linalg.cholesky(BigC_corr)
        
    def _compute_drifts_slice(self, F_ois, V_ois, V_eur, slice_idx):
        taus_slice = self.tau[slice_idx:]
        
        gamma_ois = (taus_slice * V_ois) / (1.0 + taus_slice * F_ois)
        
        corr_slice = self.drift_corr_matrix[slice_idx:, slice_idx:]
        
        sum_term = gamma_ois @ corr_slice.T
        
        drift_ois = -V_ois * sum_term
        drift_eur = -V_eur * self.rho_ois_eur * sum_term
        
        return drift_ois, drift_eur

    def simulate(self, M_steps=SIM_STEPS, n_paths=SIM_PATHS, seed=SEED, dW_custom=None):
        dt = self.T[-1] / M_steps
        sq_dt = torch.sqrt(dt)
        
        if dW_custom is None:
            # standard pseudo-random numbers
            gen = torch.Generator(device=self.device).manual_seed(seed)
            # uncorrelated shocks
            Z = torch.randn(n_paths, M_steps, 4*self.N, device=self.device, generator=gen)
            
            # correlate
            dW = (Z @ self.L.T) * sq_dt
        else:
            dW = dW_custom

        # initialization
        # expand => stretch shape but does not copy in memory
        hist_F_ois = [self.F0_ois.expand(n_paths, -1)]
        hist_F_eur = [self.F0_eur.expand(n_paths, -1)]
        
        curr_fo = hist_F_ois[0]
        curr_fe = hist_F_eur[0]
        curr_vo = self.alphas.expand(n_paths, -1)
        curr_ve = self.alphas.expand(n_paths, -1)

        vol_drift_correction = -0.5 * self.nus**2 * dt
        time_grid = torch.linspace(0, self.T[-1], M_steps + 1, device=self.device)
        
        # time loop
        for t in range(M_steps):
            t_curr = time_grid[t]
            start_idx = int(torch.sum(self.T[:self.N] <= t_curr).item())
            
            if start_idx >= self.N:
                hist_F_ois.append(curr_fo)
                hist_F_eur.append(curr_fe)
                continue

            # slice active state
            fo = curr_fo[:, start_idx:]
            fe = curr_fe[:, start_idx:]
            vo = curr_vo[:, start_idx:]
            ve = curr_ve[:, start_idx:]
            
            dW_fo = dW[:, t, 0 + start_idx : self.N]
            dW_fe = dW[:, t, self.N + start_idx : 2*self.N]
            dW_vo = dW[:, t, 2*self.N + start_idx : 3*self.N]
            dW_ve = dW[:, t, 3*self.N + start_idx : 4*self.N]
            
            curr_nus = self.nus[start_idx:]
            drift_corr = vol_drift_correction[start_idx:]

            # update vol
            vo_next = vo * torch.exp(drift_corr + curr_nus * dW_vo)
            ve_next = ve * torch.exp(drift_corr + curr_nus * dW_ve)

            # Predictor-Corrector drifts
            mu_ois, mu_eur = self._compute_drifts_slice(fo, vo, ve, start_idx)
            
            # predictor
            fo_pred = fo + mu_ois*dt + vo*dW_fo
            
            # corrector
            mu_ois_p, mu_eur_p = self._compute_drifts_slice(fo_pred, vo, ve, start_idx)
            mu_ois_avg = 0.5 * (mu_ois + mu_ois_p)
            mu_eur_avg = 0.5 * (mu_eur + mu_eur_p)
            
            # update rate
            fo_next = fo + mu_ois_avg*dt + vo*dW_fo
            fe_next = fe + mu_eur_avg*dt + ve*dW_fe
            
            # reconstruct full tensor
            if start_idx > 0:
                # concta => cat
                curr_fo = torch.cat([curr_fo[:, :start_idx], fo_next], dim=1)
                curr_fe = torch.cat([curr_fe[:, :start_idx], fe_next], dim=1)
                curr_vo = torch.cat([curr_vo[:, :start_idx], vo_next], dim=1)
                curr_ve = torch.cat([curr_ve[:, :start_idx], ve_next], dim=1)
            else:
                curr_fo = fo_next
                curr_fe = fe_next
                curr_vo = vo_next
                curr_ve = ve_next

            hist_F_ois.append(curr_fo)
            hist_F_eur.append(curr_fe)

        return time_grid, torch.stack(hist_F_ois, dim=1), torch.stack(hist_F_eur, dim=1)

    def get_ois_df_to_step(self, paths_ois, target_step):
        """ discount factor P(0, T_step) pathwise  """
        
        if target_step == 0:
            # if t= -> P=1
            return torch.ones(paths_ois.shape[0], device=self.device)

        # paths_ois: (n_paths, n_steps + 1, n_tenors)
        M_steps = paths_ois.shape[1] - 1
        dt = self.T[-1] / M_steps
        
        time_grid = torch.linspace(0, self.T[-1], M_steps + 1, device=self.device)
        relevant_times = time_grid[:target_step]
        
        # find first tenor T_k such that T_k > t_current
        # => mask: (target_step, N)
        mask = self.T[None, :] > relevant_times[:, None]
        
        # index of the first True value (the active tenor)
        spot_indices = torch.argmax(mask.int(), dim=1)
        
        # relevant spot rates from the path
        # paths_ois sliced: (Paths, target_step, Tenors)
        paths_slice = paths_ois[:, :target_step, :]
        
        # expand indices to (Paths, target_step, 1) for torch.gather
        expanded_indices = spot_indices[None, :, None].expand(paths_ois.shape[0], -1, -1)

        # gather along the tenor dimension (dim=2) using spot_indices
        # result: (Paths, target_step, 1) -> squeeze to (Paths, target_step)
        spot_rates = torch.gather(paths_slice, 2, expanded_indices).squeeze(-1)
        
        # discount factor
        step_dfs = 1.0 / (1.0 + spot_rates * dt)
        
        # cumulative product across time steps
        total_df = torch.prod(step_dfs, dim=1)
        
        return total_df
    
    def get_terminal_bond(self):
        """ P(0, T_N) using  initial F0_ois parameters"""
        dfs = 1.0 / (1.0 + self.tau * self.F0_ois)
        return torch.prod(dfs)


