import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, Any


class FastEquilibriumSolver:
    """
    A solver for calculating market equilibrium prices and recovering marginal costs
    using a Nested Logit demand model.

    Includes:
    1. Robust 'load_market' that aligns indices based on 'j'.
    2. Log-Sum-Exp trick in 'compute_shares_and_jac' for numerical stability.
    """

    def __init__(self, df_dem: pd.DataFrame, df_sup: pd.DataFrame, use_log_price: bool = True):
        """
        Initialize the solver.

        Args:
            df_dem: Demand-side dataframe (households).
            df_sup: Supply-side dataframe (products).
            use_log_price: If True, utility uses alpha * log(P). If False, alpha * P.
        """
        self.df_dem = df_dem
        self.df_sup = df_sup
        self.use_log_price = use_log_price
        self.cache = {}

    def load_market(self, t_str: str) -> Dict[str, Any]:
        """
        Prepare data matrices for one time period (market).
        Key step: ensure alignment between supply-side product sorting
        and demand-side choice rows.
        """
        if t_str in self.cache:
            return self.cache[t_str]

        # 1) Supply side: filter and sort by product id j.
        s_t = self.df_sup[self.df_sup['t_str'] == t_str].copy()
        s_t = s_t.sort_values('j').reset_index(drop=True)

        # 2) Demand side: filter and sort by household/product.
        d_t = self.df_dem[self.df_dem['t_str'] == t_str].copy()
        d_t = d_t.sort_values(['h', 'j']).reset_index(drop=True)

        # 3) Ownership matrix Omega: same-brand products belong to same firm.
        firms = s_t['brand'].values
        Omega = (firms[:, None] == firms[None, :]).astype(float)

        data = {
            'Omega': Omega,
            'd_t': d_t,
            's_t': s_t,
        }
        self.cache[t_str] = data
        return data

    def compute_shares_and_jac(
        self,
        p_curr: np.ndarray,
        subsidy_curr: np.ndarray,
        data: Dict[str, Any],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute market shares (Pandas pipeline) and Jacobian (NumPy vectorization).
        """

        # --- 1) Share calculation (DataFrame logic) ---
        wdf = data['d_t'].copy()

        J = len(data['d_t'].j.unique())
        H = len(data['d_t'].h.unique())

        # Broadcast product-level vectors to long-table order: (J,) -> (H*J,).
        price_vec = np.tile(p_curr, H)
        subsidy_vec = np.tile(subsidy_curr, H)
        wdf['net_price'] = np.maximum(price_vec - subsidy_vec, 1e-6)

        # A. Utility
        if self.use_log_price:
            wdf['util_final'] = wdf['base_utility'] + wdf['alpha'] * np.log(wdf['net_price'])
        else:
            wdf['util_final'] = wdf['base_utility'] + wdf['alpha'] * wdf['net_price']

        # B. Inclusive value term: V / (1-rho)
        wdf['v_scaled'] = wdf['util_final'] / (1 - wdf['rho'])

        # C. Log-Sum-Exp trick inside each (h, nest)
        wdf['v_max'] = wdf.groupby(['h', 'nesting_ids'])['v_scaled'].transform('max')
        wdf['exp_v_shifted'] = np.exp(wdf['v_scaled'] - wdf['v_max'])
        wdf['sum_exp_shifted'] = wdf.groupby(['h', 'nesting_ids'])['exp_v_shifted'].transform('sum')

        # D. Conditional probability P(j|g)
        wdf['prob_j_g'] = wdf['exp_v_shifted'] / wdf['sum_exp_shifted']

        # E. Nest probability P(g)
        # log(D_g) = (1-rho) * (log(sum_exp_shifted) + v_max)
        wdf['log_D_g'] = (1 - wdf['rho']) * (np.log(wdf['sum_exp_shifted']) + wdf['v_max'])
        wdf['D_g'] = np.exp(wdf['log_D_g'])

        # Build denominator per household on unique nest rows.
        nest_level = wdf[['h', 'nesting_ids', 'D_g']].drop_duplicates()
        nest_level['Sum_D_g'] = nest_level.groupby('h')['D_g'].transform('sum')

        # Map denominator back to product-level rows.
        wdf = wdf.merge(nest_level[['h', 'nesting_ids', 'Sum_D_g']], on=['h', 'nesting_ids'])

        # P(g) = D_g / (1 + sum_g D_g)
        wdf['prob_g'] = wdf['D_g'] / (1 + wdf['Sum_D_g'])

        # F. Final household-product choice probability s_jht
        wdf['s_jht'] = wdf['prob_j_g'] * wdf['prob_g']

        # Aggregate weighted market shares for optimizer usage.
        shares = np.sum(
            wdf['h_weight'].values.reshape(H, J) * wdf['s_jht'].values.reshape(H, J),
            axis=0,
        )

        # --- 2) Jacobian calculation (vectorized NumPy) ---
        # Convert long table into matrices of shape (H, J).
        S_mat = wdf['s_jht'].values.reshape(H, J)       # s_j
        Sjg_mat = wdf['prob_j_g'].values.reshape(H, J)  # s_j|g
        alpha_mat = wdf['alpha'].values.reshape(H, J)
        rho_vec = wdf['rho'].values.reshape(H, J)[:, 0:1]  # (H, 1)
        net_p_mat = wdf['net_price'].values.reshape(H, J)

        # Nest ids matrix for within-nest mask.
        nest_mat = wdf['nesting_ids'].values.reshape(H, J)

        # Prepare broadcast tensors with target shape (H, J, J).
        S_i = S_mat[:, :, None]      # s_j (H, J, 1)
        S_k = S_mat[:, None, :]      # s_k (H, 1, J)
        Sjg_k = Sjg_mat[:, None, :]  # s_k|g (H, 1, J)

        # Structural masks.
        Mask_Diag = np.eye(J)[None, :, :]  # indicator(j == k)
        Mask_Nest = (nest_mat[:, :, None] == nest_mat[:, None, :]).astype(float)  # indicator(g_j == g_k)

        # Price sensitivity dV_k / dp_k.
        if self.use_log_price:
            dV_dp = alpha_mat / net_p_mat  # d(alpha*log(p))/dp = alpha/p
        else:
            dV_dp = alpha_mat              # d(alpha*p)/dp = alpha

        AP_k = dV_dp[:, None, :]  # (H, 1, J), broadcast over j

        # Bracket term:
        # [ (1/(1-rho))*1{j=k} - (rho/(1-rho))*s_k|g*1{g_j=g_k} - s_k ]
        rho_term = rho_vec[:, :, None]  # (H, 1, 1)
        inv_rho_comp = 1.0 / (1.0 - rho_term)
        rho_ratio = rho_term * inv_rho_comp

        Term1 = Mask_Diag * inv_rho_comp          # own-price diagonal effect
        Term2 = Mask_Nest * rho_ratio * Sjg_k     # within-nest substitution
        Term3 = S_k                                # market-wide substitution

        Bracket = Term1 - Term2 - Term3

        # d(s_j)/d(p_k) = s_j * Bracket * (dV_k/dp_k)
        Jac_h = S_i * Bracket * AP_k

        # Weighted household aggregation -> market-level Jacobian (J, J).
        weights_mat = np.array(list(data['d_t'].groupby('h')['h_weight'].first())).reshape(H, 1, 1)
        Jacobian = np.sum(weights_mat * Jac_h, axis=0)

        return shares, Jacobian

    def recover_mc(self, t_str: str) -> np.ndarray:
        """
        Invert first-order conditions to recover marginal costs:
        MC = P - (Omega * Delta)^-1 * s
        """
        data = self.load_market(t_str)

        p_obs = data['s_t']['price'].values.astype(float)
        subsidy_obs = data['s_t']['subsidy'].values.astype(float)
        shares, Jacobian = self.compute_shares_and_jac(p_obs, subsidy_obs, data)
        Lambda = data['Omega'] * Jacobian

        try:
            markup = np.linalg.solve(Lambda, -shares)
        except np.linalg.LinAlgError:
            # Regularize if the system is singular.
            markup = np.linalg.solve(Lambda + np.eye(len(p_obs)) * 1e-6, -shares)

        mc = p_obs - markup
        return mc

    def solve_price(
        self,
        t_str: str,
        mc_vec: Optional[np.ndarray] = None,
        tol: float = 1e-4,
        max_iter: int = 20,
        price_upper_factor: float = 3.0,
    ) -> Optional[pd.DataFrame]:
        """
        Equilibrium solver using a damped fixed-point / diagonal Jacobian approximation.
        Update rule: p_new = p - F(p) / 2.
        """
        # 1) Load data
        data = self.load_market(t_str)
        subsidy = data['s_t']['subsidy'].values.astype(float)
        p_curr = data['s_t']['price'].values.astype(float)
        subsidy = np.minimum(subsidy, 0.6 * p_curr)

        mc = mc_vec if mc_vec is not None else data['s_t']['mc'].values.astype(float)
        Omega = data['Omega']

        n = len(p_curr)
        p_min = mc * 1.05
        p_max = mc * price_upper_factor

        err = 1e6
        eye_n = np.eye(n)  # for regularized linear solves

        for it in range(max_iter):
            # 1) Compute shares and full Jacobian
            shares, Jac = self.compute_shares_and_jac(p_curr, subsidy, data)

            # 2) Markup solve and residual
            Lambda = Omega * Jac
            try:
                markup = np.linalg.solve(Lambda, -shares)
            except np.linalg.LinAlgError:
                markup = np.linalg.solve(Lambda + eye_n * 1e-4, -shares)

            F = p_curr - (mc + markup)
            err = np.max(np.abs(F))
            if err < tol:
                break

            # 3) Damped update (stable heuristic for this Nested Logit setup)
            delta = -F / 2.0
            p_new = np.clip(p_curr + delta, p_min, p_max)
            p_curr = p_new

        if err > tol:
            print(f"[WARN] {t_str}: did not reach tol (err={err:.3e})")

        return pd.DataFrame({'j': data['s_t']['j'].values, 'price_equil': p_curr})
