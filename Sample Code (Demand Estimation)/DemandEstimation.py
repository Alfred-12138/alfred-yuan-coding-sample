
# %%
import numpy as np
import pandas as pd
import pyhdfe
import matplotlib.pyplot as plt
import os
import sys
import warnings
from scipy.optimize import minimize

def fast_2sls(y, X, Z):
    """Efficient 2SLS Estimator: Beta = (X_hat' X_hat)^-1 X_hat' y"""
    try:
        # 1. First Stage: Regress X on Z -> X_hat
        gamma, _, _, _ = np.linalg.lstsq(Z, X, rcond=None)
        X_hat = Z @ gamma
        # 2. Second Stage: Regress y on X_hat
        beta, _, _, _ = np.linalg.lstsq(X_hat, y, rcond=None)
        return beta.flatten()
    except:
        return np.zeros(X.shape[1])

import scipy.stats as stats


def calculate_gmm_se_robust(X, Z, y, beta, n_groups_to_adjust=0):
    """
    Computes GMM Robust Standard Errors using the explicit Sandwich Formula.
    Robust to heteroskedasticity and singular matrices (uses pinv).
    
    Formula: V = (G'WG)^-1  [G' W S W G]  (G'WG)^-1
    """
    N, K = X.shape
    
    # 1. Residuals (u)
    # y, X, beta must be aligned
    u = y - X @ beta
    
    # 2. Gradient of Moments (G)
    # g(beta) = Z'u = Z'(y - X*beta)
    # G = d(g)/d(beta) = -Z'X
    # We can ignore the negative sign as it cancels in the square quadratic form
    G = Z.T @ X 
    
    # 3. Weighting Matrix (W)
    # We used 2SLS logic, so W = (Z'Z)^-1
    # Use pinv for stability
    try:
        W = np.linalg.pinv(Z.T @ Z)
    except:
        # Fallback identity if Z is terrible (unlikely)
        W = np.eye(Z.shape[1])
        
    # 4. Score Variance Matrix (S) - The "Meat"
    # S = Sum( z_i * z_i' * u_i^2 ) = Z' * diag(u^2) * Z
    # Memory efficient implementation:
    u_sq = (u ** 2).flatten()
    # Multiply each row of Z by u (broadcasting), then dot product
    # This avoids creating the massive N x N diag matrix
    Z_weighted = Z * np.sqrt(u_sq)[:, None] 
    S = Z_weighted.T @ Z_weighted
    
    # 5. Calculate Bread: (G' W G)^-1
    # This is the sensitivity of the moment conditions to the parameters
    Bread_inv = G.T @ W @ G
    Bread = np.linalg.pinv(Bread_inv)
    
    # 6. Calculate Variance
    # V = Bread * (G' W S W G) * Bread
    # Let's verify the center part: Matrix of constraints weighted by error variance
    Meat = G.T @ W @ S @ W @ G
    
    V_robust = Bread @ Meat @ Bread
    
    # 7. Small Sample / DOF Adjustment
    # Adjust for N / (N - K - FixedEffects)
    dof_scale = N / max(1.0, (N - K - n_groups_to_adjust))
    V_robust *= dof_scale
    
    # 8. Extract Standard Errors
    # abs() covers rare negative zeros from numerical noise
    se = np.sqrt(np.abs(np.diag(V_robust)))
    
    return se

def calculate_observable_utility(df):
    """
    Calculates X * Beta (excluding Price and Xi).
    Uses the 'beta_' columns in the dataframe.
    """
    beta_cols = [col for col in df.columns if col.startswith('beta_')]
    # Added beta_is_phev logic
    # Multiply each characteristic column by its corresponding beta column and sum
    val_xbeta = sum(df[col.replace('beta_', '')] * df[col] for col in beta_cols)
    return val_xbeta


def calculate_total_utility(df, alpha_col='alpha', price_col='log_net_price'):
    """
    Reconstructs Total Deterministic Utility.
    U = Xi + Alpha * log(P) + X * Beta
    """
    # 1. Observable Characteristics (X * Beta)
    val_xbeta = calculate_observable_utility(df)

    # 2. Price Utility (Alpha * Price)
    val_price = df[alpha_col] * df[price_col]

    # 3. Total = Xi + Price + X*Beta
    return df['util_xi'] + val_price + val_xbeta


def solve_nested_logit_shares(df, utility_col):
    """
    Computes market shares given a utility column using the Log-Sum-Exp trick.
    Returns: Vector of predicted shares.
    """
    wdf = df.copy()

    # 1. Inclusive Value Calculation (V / (1-rho))
    wdf['v_scaled'] = wdf[utility_col] / (1 - wdf['rho'])

    # Log-Sum-Exp Trick for Nest Sum
    # Nests are now strictly defined by nesting_ids (BEV, PHEV, ICE)
    wdf['v_max'] = wdf.groupby(['market_ids', 'nesting_ids'])[
        'v_scaled'].transform('max')
    wdf['exp_v_shifted'] = np.exp(wdf['v_scaled'] - wdf['v_max'])
    wdf['sum_exp_shifted'] = wdf.groupby(['market_ids', 'nesting_ids'])[
        'exp_v_shifted'].transform('sum')

    # Prob(j|g)
    wdf['prob_j_g'] = wdf['exp_v_shifted'] / wdf['sum_exp_shifted']

    # 2. Group Probability
    # log(D_g) = (1-rho) * (log(sum_exp) + v_max)
    groups = wdf[['market_ids', 'nesting_ids',
                  'sum_exp_shifted', 'v_max', 'rho']].drop_duplicates()
    groups['log_D_g'] = (1 - groups['rho']) * \
        (np.log(groups['sum_exp_shifted']) + groups['v_max'])

    # Log-Sum-Exp Trick for Market Denominator
    groups['D_g'] = np.exp(groups['log_D_g'])
    market_denom = groups.groupby('market_ids')['D_g'].sum(
    ).reset_index().rename(columns={'D_g': 'Sum_D_g'})
    groups = groups.merge(market_denom, on='market_ids')

    # Prob(g) = D_g / (1 + Sum_D_g) -> Assuming outside option V_0 = 0 -> exp(0) = 1
    groups['prob_g'] = groups['D_g'] / (1 + groups['Sum_D_g'])

    wdf = wdf.merge(groups[['market_ids', 'nesting_ids', 'prob_g']], on=[
                    'market_ids', 'nesting_ids'])

    return wdf['prob_j_g'] * wdf['prob_g']



# %%
def joint_soft_solver(y, X, Z, n_common, target=-4.0, penalty=200.0, keep_mask=None):

    """
    Solves GMM for a homogeneous group.
    Assumes X structure: [Common Controls ..., Alpha, Rho]
    """
    # Weighting Matrix
    ZtZ = Z.T @ Z
    try:
        W = np.linalg.pinv(ZtZ)
    except:
        W = np.eye(Z.shape[1])
        
    Zty = Z.T @ y
    ZtX = Z.T @ X
    
    # --- 1. Construct Bounds Dynamically ---
    # We map the reduced X back to logical variables.
    # Assumed Logical Order: [Common_1, ..., Common_k, Alpha, Rho]
    
    bounds = []
    logical_idx = 0 
    
    # A. Common Variables (Unbounded)
    for i in range(X.shape[1] - 2):
        bounds.append((-np.inf, np.inf))
    
    # Last 2 bounds: Alpha (-inf, 0) and Rho (0, 1)
    bounds.append((-np.inf, 0))
    bounds.append((0, 1))

    # --- 2. Objective Function ---
    def objective(beta):
        # A. GMM Fit: (y - Xb)' Z W Z' (y - Xb)
        # Simplified to: g' W g where g = Z'y - Z'Xb
        g = Zty - ZtX @ beta
        loss_gmm = g.T @ W @ g
        
        
        return loss_gmm 

    # --- 3. Run Optimization ---
    beta_init = np.zeros(X.shape[1])
    
    res = minimize(
        objective, 
        beta_init, 
        method='L-BFGS-B', 
        bounds=bounds,
        options={'gtol': 1e-4, 'ftol': 1e-9}
    )
    
    return res.x
# %%
