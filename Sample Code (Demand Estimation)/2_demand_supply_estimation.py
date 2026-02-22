# %%
import numpy as np
import pandas as pd
import pyhdfe
import matplotlib.pyplot as plt
import os
import sys
import warnings

# Add code directory to path to import the solver
sys.path.append(os.path.join(os.path.dirname(__file__), "code"))
sys.path.append(os.path.join(os.path.dirname(__file__)))
from EquilibriumSolver import FastEquilibriumSolver
from DemandEstimation import fast_2sls, calculate_gmm_se_robust, joint_soft_solver

warnings.simplefilter('ignore')

print("\n" + "="*80)
print("STRUCTURAL ESTIMATION PIPELINE: AUTOMOTIVE MARKET")
print("Model: Nested Logit")
print("Features: Distinct Cost Functions")
print("="*80)



# %%
# ==============================================================================
# SECTION A: DATA PIPELINE
# ==============================================================================

# [0] Load Data
# ------------------------------------------------------------------------------
print("[0] Loading & Preprocessing Data...")
data_path = os.path.join(os.path.dirname(__file__), "..", "data")
output_path = os.path.join(os.path.dirname(__file__), "..", "fig")
# Adjust filename as needed

final_sales = pd.read_csv(os.path.join(data_path, 'final_sales_panel_annual.csv'))
final_attrs = pd.read_csv(os.path.join(data_path, 'final_model_attributes.csv'))

# Merge final_attrs into final_sales on Models column
df = final_sales.merge(final_attrs, on=['Models','Year', 'Geo_Market'], how='left')
del final_sales, final_attrs

# %%
# Fill missing values by model (forward/backward fill within each model)
print("[0.5] Filling Missing Values by Model and Year...")
fill_cols = ['FuelType', 'Brand','Manufacturer', 'Maxpower', 'Displacement', 
             'SizeSegment', 'BodyType', 'Length', 'Width', 'Height', 'FuelEfficiency']

for col in fill_cols:
    if col in df.columns:
        # First: Fill missing values using same Year + Model average
        if df[col].dtype == 'object':
            # Categorical: use mode within Year-Model group
            df[col] = df.groupby(['Year', 'Models'])[col].transform(
                lambda x: x.fillna(x.mode()[0] if len(x.mode()) > 0 else x.iloc[0])
            )
        else:
            # Numeric: use mean within Year-Model group
            df[col] = df.groupby(['Year', 'Models'])[col].transform(
                lambda x: x.fillna(x.mean())
            )
        
        # Second: If still NaN, use same Model (across all years) mode/mean
        if df[col].isna().any():
            if df[col].dtype == 'object':
                df[col] = df.groupby('Models')[col].transform(
                    lambda x: x.fillna(x.mode()[0] if len(x.mode()) > 0 else x.iloc[0])
                )
            else:
                df[col] = df.groupby('Models')[col].transform(
                    lambda x: x.fillna(x.mean())
                )
        
        # Third: If still NaN (e.g., single model with all NaN), use global mode/mean
        if df[col].isna().any():
            if df[col].dtype == 'object':
                df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown', inplace=True)
            else:
                df[col].fillna(df[col].mean(), inplace=True)

print(f"   -> Missing values filled. Remaining NaNs: {df[fill_cols].isna().sum().sum()}")
# df.isna().sum()


# %%
# The columns are as follows:
# print(df.columns)
# ['Year', 'Geo_Market', 'Models', 'Quantity', 'Avg_Price',
#'Population_Million', 'Market_Size_Households', 'Brand', 'Manufacturer',
# 'FuelType', 'Maxpower', 'Displacement', 'SizeSegment', 'BodyType',
# 'Length', 'Width', 'Height', 'FuelEfficiency']
# df.isna().sum()

# Year                           0
# Geo_Market                     0
# Models                         0
# Quantity                       0
# Avg_Price                      0
# Population_Million             0
# Market_Size_Households         0
# Brand                      36505
# Manufacturer               36505
# FuelType                   36823
# Maxpower                   85935
# Displacement               36826
# SizeSegment                36593
# BodyType                   36505
# Length                     38306
# Width                      38306
# Height                     38306
# FuelEfficiency            159004
# market_ids                     0
# shares                         0
# price                          0
# market_segment                 0

# Map fuel types to ICEV, BEV, PHEV
fuel_type_map = {
    '汽油': 'ICEV',           # Gasoline
    '柴油': 'ICEV',           # Diesel
    '甲醇': 'ICEV',           # Methanol
    '燃料电池': 'ICEV',       # Fuel Cell (treated as ICEV for now)
    '纯电动': 'BEV',          # Battery Electric
    '混合动力': 'PHEV',       # Hybrid
    '插电式': 'PHEV'          # Plug-in Hybrid
}

df['fuel_type'] = df['FuelType'].map(fuel_type_map).fillna('ICEV')
# %%

print("[0.55] Checking unique model counts by fuel type...")
print('ICEV', df.loc[df['fuel_type']=='ICEV','Models'].nunique())
print('BEV', df.loc[df['fuel_type']=='BEV','Models'].nunique())
print('PHEV', df.loc[df['fuel_type']=='PHEV','Models'].nunique())
# %%
# Check and fill EV-specific columns for ICEV vehicles
print("[0.6] Filling EV-specific columns for ICEV vehicles...")
ev_specific_cols = ['ElectricityRange', 'ElectricMotorPower', 'BatteryCapacity_kWh', 'ElectricEfficiency']

for col in ev_specific_cols:
    if col in df.columns:
        # Count missing values before
        nan_count_before = df[col].isna().sum()
        
        # For ICEV vehicles, fill EV-specific columns with 0
        df.loc[df['fuel_type'] == 'ICEV', col] = 0
        
        # Count missing values after
        nan_count_after = df[col].isna().sum()
        
        print(f"   -> {col:<25} : {nan_count_before:>6} NaN -> {nan_count_after:>6} NaN (Filled {nan_count_before - nan_count_after} ICEV rows with 0)")
    else:
        print(f"   -> {col:<25} : Column not found in dataframe") 

# %%
# Fill remaining missing values for other columns
print("[0.7] Filling remaining missing values...")

# Fill remaining numeric columns with fuel_type-specific median
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if df[col].isna().any():
        nan_before = df[col].isna().sum()
        # Fill with fuel_type-specific median
        df[col] = df.groupby('fuel_type')[col].transform(
            lambda x: x.fillna(x.median())
        )
        # If still NaN (fuel type group all NaN), use global median
        if df[col].isna().any():
            df[col].fillna(df[col].median(), inplace=True)
        nan_after = df[col].isna().sum()
        if nan_before > 0:
            print(f"   -> {col:<25} : {nan_before:>6} NaN -> {nan_after:>6} NaN (Filled with fuel_type median)")

print(f"   -> All remaining NaNs filled. Total NaNs in dataframe: {df.isna().sum().sum()}")

# %%


# %%
# [1] Preprocessing
# ------------------------------------------------------------------------------
df.columns
# ID Management

df['market_ids'] = df['Year'].astype(str) + '-' + df['Geo_Market']
df['shares'] = df['Quantity']  / df['Market_Size_Households']
df['price'] = df['Avg_Price'] / 10000

# Filter positive shares

# Market Segment
def assign_market_segment(price):
    if price <= 12:
        return 'entry'
    elif price <= 20:
        return 'mainstream'
    else:
        return 'premium'

df['market_segment'] = df['price'].apply(assign_market_segment)

# --- [MODIFICATION START] Nesting & Flags ---
# Distinct flags for Fuel Types
df['is_bev'] = (df['fuel_type'] == 'BEV').astype(int)
df['is_phev'] = (df['fuel_type'] == 'PHEV').astype(int)
df['is_ice'] = (~df['fuel_type'].isin(['BEV', 'PHEV'])).astype(int)


# --- [MODIFICATION END] ---

# %%
# Regressors Construction
# ------------------------------------------------------------------------------

# Tech Features
df['log_size'] = np.log(df['Length'] * df['Width'] * df['Height'])

# EV Range: Applies to BEV and PHEV (if data exists), but mostly BEV
# We multiply by (is_bev + is_phev) to ensure ICE gets 0
df['log_ev_range'] = np.log(
    df['ElectricityRange'].fillna(0) + 1) * (df['is_bev'] + df['is_phev'])
df['log_ev_power'] = np.log(  
    df['ElectricMotorPower'].fillna(0) + 1) * (df['is_bev'] + df['is_phev'])
df['log_battery_capacity'] = np.log(
    df['BatteryCapacity_kWh'].fillna(0) + 1) * (df['is_bev'] + df['is_phev'])
df['electric_efficiency'] = df['ElectricEfficiency'].fillna(0) * (df['is_bev'] + df['is_phev'])

# Fuel Consumption: Applies to ICE and PHEV
df['fuel_consumption'] = df['FuelEfficiency'].fillna(0) * (df['is_ice'] + df['is_phev'])
df['displacement'] = df['Displacement'].fillna(0) * (df['is_ice'] + df['is_phev'])
df['fuel_efficiency'] = df['FuelEfficiency'].fillna(0) * (df['is_ice'] + df['is_phev'])
df['max_power'] = df['Maxpower'].fillna(0)

df['car_class'] = df['BodyType'] 
df['size_segment']= df['SizeSegment']
# %%

columns_to_keep = ['market_ids', 'Models','Year','Brand', 'Geo_Market','car_class','fuel_type', 'size_segment', 'shares', 'price',
                   'log_size', 'log_ev_range', 'log_ev_power', 'log_battery_capacity',
                     'electric_efficiency', 'fuel_consumption', 'displacement',
                        'fuel_efficiency', 'max_power']

df = df[columns_to_keep].copy()
# Log Variables for Regression
df['s_0'] = 1 - df.groupby('market_ids')['shares'].transform('sum')
df = df[df['s_0'] > 1e-4].copy()  # Safety filter


# %%
# 你的原始数据
brands_list = np.array(['奇瑞', '大众', '日产', 'DS', '比亚迪', '吉利', '红旗', '现代', '纳智捷', 'MG', '丰田',
       '凯迪拉克', '东南', '斯柯达', '雪铁龙', '中华', '雪佛兰', '众泰', '传祺', '本田', '起亚',
       '凯翼', '别克', '铃木', '力帆', '三菱', '北汽制造', '北京', '北汽昌河', '启辰', '江淮',
       '哈弗', '奔驰', '福特', '华泰', '夏利', '大通', '长安', '一汽', '东风', '吉奥', '奥迪',
       '宝马', '宝骏', '思铭', '黄海', '海马', '标致', '永源', '沃尔沃', '长城', '猎豹', '理念',
       '马自达', '莲花', '腾势', '菲亚特', '英菲尼迪', '荣威', '福田', '观致', '金杯', '陆风',
       '江铃', '中兴', '北汽幻速', '开瑞', '福迪', '知豆', '五十铃', '英致', '野马', '卡威',
       '双环', '江南', '哈飞', '新凯', '克莱斯勒', '红星', '之诺', '华颂', '路虎', '吉普', '迈迪',
       '宝沃', '五菱', '北汽威旺', '捷豹', '斯威', '汉腾', '雷诺', '九龙', '讴歌', '启腾', '比速',
       '御捷', '欧联', '云度', '华骐', '电咖', '领克', 'WEY', '成功', '国金', '通家福', '裕路',
       '陆地方舟', '君马', '大乘', '威马', '小鹏', '捷途', '广汽', '蔚来', '领途', '哪吒', '开沃',
       '欧拉', '新特', '前途', '云雀', '恒润', '瑞驰', '埃安', '特斯拉', '吉利几何', '捷达',
       '汉龙', '星途', '爱驰', '理想', '潍柴', '零跑', '大运', '国机智骏', '雷丁', 'ARCFOX',
       '道达', '赛麟', '速达', '林肯', '恒天', '赛力斯', '比德文', '高合', '合创', '坦克', '天际',
       '极星', '睿蓝', '凌宝', '创维', '摩登', '岚图', '智己', '极氪', '朋克', '电动屋', '小虎',
       '新日', '百智', '北汽瑞翔', '松散机车', '鸿蒙智行', 'SMART', '恒驰', '阿维塔', '飞凡',
       '华梓', '敏安', '仰望', '昊铂', '极石', '极越', '猛士', '蓝电', '方程豹', '路特斯',
       '吉利银河', 'ICAR', '远航', '未奥', 'MINI', '乐道', '小米', '鑫源'])

def get_brand_group(brand_name):
    # ==========================================
    # 1. JV / Foreign Luxury (合资/外资豪华)
    # ==========================================
    # 包括 BBA, 特斯拉, 二线豪华
    if brand_name in [
        '奔驰', '宝马', '奥迪', '凯迪拉克', '沃尔沃', '路虎', '捷豹', 
        '林肯', '英菲尼迪', '讴歌', 'DS', '特斯拉', '路特斯', '极星', 
        'SMART', 'MINI', '克莱斯勒', '阿尔法·罗密欧'
    ]:
        return 'ModelsV_Luxury'

    # ==========================================
    # 2. JV Mass 
    # ==========================================
    # Traditional joint ventures and mass-market foreign brands
    elif brand_name in [
        '大众', '丰田', '本田', '日产', '别克', '雪佛兰', '福特', 
        '现代', '起亚', '马自达', '标致', '雪铁龙', '斯柯达', '铃木', 
        '三菱', '菲亚特', '雷诺', '吉普', '五十铃', '捷达', 
        '思铭', '理念', '之诺', '华骐'
    ]:
        return 'ModelsV_Mass'

    # ==========================================
    # 3. Domestic NEV & Tech Startups
    # ==========================================
    # Tech Startups
    elif brand_name in [
        '蔚来', '小鹏', '理想', '小米', '哪吒', '零跑', '威马', 
        '鸿蒙智行', '赛力斯', 'AITO', '问界',
        '极氪', '阿维塔', '智己', '埃安', '岚图', '深蓝', '飞凡', 
        '昊铂', '极越', '高合', '乐道', '方程豹', '仰望', '猛士',
        '合创', '爱驰', '天际', '云度', '前途', '创维', '极石', 
        'ARCFOX', '恒驰', '自游家', '电咖', '新特'
    ]:
        return 'Domestic_Startup_NEV'

    # ==========================================
    # 4. Domestic Legacy
    # ==========================================

    else:

        return 'Domestic_Legacy'


# %%

df['brand_group'] = df['Brand'].apply(get_brand_group)
# Create dummies for car_class
df['is_suv'] = df['car_class'].apply(lambda x: 1 if x=='SUV' else 0)
df['is_mpv'] = df['car_class'].apply(lambda x: 1 if x=='MPV' else 0)

conditions = [
    df['fuel_type'] == 'BEV',
    df['fuel_type'] == 'PHEV'
]
# choices = ['BEV', 'PHEV']
# 🔧 MODIFICATION: nesting_ids based on size_segment
df['nesting_ids'] = df['brand_group']


grp = df.groupby(['market_ids', 'nesting_ids'])
df['nest_sum_share'] = grp['shares'].transform('sum')
df['within_nest_share'] = df['shares'] / df['nest_sum_share']

# Smoothing
df['nest_count'] = grp['nesting_ids'].transform('count')
df['within_nest_share_smoothed'] = np.where(
    df['nest_count'] == 1, 0.999, df['within_nest_share'])

df['log_share_ratio'] = np.log(df['shares'] / df['s_0'])
df['log_net_price'] = np.log(df['price']) # 🔧 MODIFICATION NEEDED: Review net_price calculation
df['log_within_share'] = np.log(df['within_nest_share_smoothed'])

# Cleaning
req_cols = ['log_share_ratio', 'log_net_price', 'log_within_share',
            'max_power', 'log_ev_range', 'fuel_consumption']
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(subset=req_cols, inplace=True)

# %%
# ==============================================================================
# [2] Constructing Improved Instruments (IVs) - Refined & Pruned
# ==============================================================================
import scipy.spatial.distance as dist
import numpy as np
import pandas as pd

print("[2] Constructing Improved Instruments (High Quality Only)...")

# --- Helper Functions ---

def build_differentiation_ivs(df, char, grp_cols):
    """
    Differentiation IVs: Measures isolation in characteristic space.
    """
    # 1. Sum X
    sum_x = df.groupby(grp_cols)[char].transform('sum')
    sum_x_rival = sum_x - df[char]
    
    # 2. Sum X^2
    df['temp_sq'] = df[char] ** 2
    sum_sq = df.groupby(grp_cols)['temp_sq'].transform('sum')
    sum_sq_rival = sum_sq - df['temp_sq']
    
    # 3. Count
    count = df.groupby(grp_cols)[char].transform('count')
    count_rival = count - 1
    
    # 4. Formula
    diff_iv = (count_rival * df['temp_sq']) - (2 * df[char] * sum_x_rival) + sum_sq_rival
    df.drop(columns=['temp_sq'], inplace=True)
    return diff_iv

def calculate_local_rivals(df, char_cols, radius=1.0):
    """
    Radius IVs: Count rivals within Z-score distance.
    """
    # Standardize cols
    z_cols = []
    for c in char_cols:
        z_col = f'z_{c}'
        df[z_col] = (df[c] - df[c].mean()) / df[c].std()
        z_cols.append(z_col)
    
    local_counts = np.zeros(len(df))
    
    # Loop by market time
    for t_val in df['Year'].unique():
        mask = df['Year'] == t_val
        if mask.sum() < 2: continue
        
        X_mat = df.loc[mask, z_cols].values
        dists = dist.cdist(X_mat, X_mat, metric='euclidean')
        
        # Exclude self (dist > 0)
        close_rivals = ((dists < radius) & (dists > 1e-6)).sum(axis=1)
        local_counts[mask] = close_rivals
        
    df.drop(columns=z_cols, inplace=True)
    return local_counts


def calculate_local_rivals_within_nest(df, char_cols, nest_col, radius=1.0):
    """
    Differentiation IV (Gandhi & Houde, 2019): 
    Count rivals that are within Z-score distance AND in the same nest.
    
    Args:
        df: DataFrame
        char_cols: List of characteristics (e.g., ['log_size', 'max_power'])
        nest_col: The column name for the nest (e.g., 'fuel_type' or 'class_segment')
        radius: The Euclidean distance threshold (in std devs)
    """
    # 1. Standardize characteristics globally
    z_cols = []
    for c in char_cols:
        z_col = f'z_{c}'
        # Handle potential zero std deviation
        std_val = df[c].std()
        if std_val == 0:
            df[z_col] = 0
        else:
            df[z_col] = (df[c] - df[c].mean()) / std_val
        z_cols.append(z_col)
    
    # Initialize result array
    local_counts = np.zeros(len(df))
    
    # 2. Loop by Market Time (t)
    for t_val in df['Year'].unique():
        # Optimization: Pre-filter for time to reduce index lookups
        t_mask = df['Year'] == t_val
        if t_mask.sum() < 2: continue
        
        # 3. Loop by Nest (g) - THIS IS THE KEY CHANGE
        # We only calculate distances among products in the same nest group
        current_nests = df.loc[t_mask, nest_col].unique()
        
        for g_val in current_nests:
            # Create a combined mask: Same Time AND Same Nest
            nest_mask = (df['Year'] == t_val) & (df[nest_col] == g_val)
            
            # Need at least 2 products in the nest to have rivals
            if nest_mask.sum() < 2: continue
            
            # Extract characteristic matrix for this nest
            X_mat = df.loc[nest_mask, z_cols].values
            
            # Compute Euclidean distance matrix
            dists = dist.cdist(X_mat, X_mat, metric='euclidean')
            
            # Count close rivals: distance < radius AND distance > 0 (exclude self)
            # 1e-6 handles floating point noise for "distance > 0"
            close_rivals = ((dists < radius) & (dists > 1e-6)).sum(axis=1)
            
            # Assign results back to the main array
            local_counts[nest_mask] = close_rivals
        
    # Clean up temporary Z-score columns
    df.drop(columns=z_cols, inplace=True)
    
    return local_counts
# %%
#  Define Characteristics
# -------------------------------------------------------------------------

iv_chars = ['max_power', 'log_size', 'log_ev_range', 'log_ev_power', 
            'log_battery_capacity']


# print(f"   -> Key Characteristics: {iv_chars}")

# -------------------------------------------------------------------------
# 2. Local Competition IVs (The "Stars")
# -------------------------------------------------------------------------

print("   -> Computing Radius IVs (Strongest Identification)...")
space_cols = ['max_power', 'log_size', 'log_ev_range', 'log_ev_power', 
            'log_battery_capacity']
# if 'log_ev_range' in iv_chars: space_cols.append('log_ev_range')

df['iv_loc_count_01'] = calculate_local_rivals(df, space_cols, radius=0.1)
df['iv_loc_count_05'] = calculate_local_rivals(df, space_cols, radius=0.5)
df['iv_loc_count_10'] = calculate_local_rivals(df, space_cols, radius=1.0)
df['iv_loc_count_nest_05'] = calculate_local_rivals_within_nest(
    df, space_cols, 'nesting_ids', radius=0.5
)
df['iv_loc_count_nest_10'] = calculate_local_rivals_within_nest(
    df, space_cols, 'nesting_ids', radius=1.0
)

# Nest Structure IVs (Standard BLP/Ghandi-Houde)
# -------------------------------------------------------------------------
# Count Rivals in Nest
# 移除 brand_group，只看 nesting_ids (EV vs EV)
df['iv_count_nest'] = df.groupby(['market_ids', 'nesting_ids'])['Models'].transform('count') - 1

df['iv_count_brand_group'] = df.groupby(['market_ids', 'brand_group'])['Models'].transform('count') - 1

# Sum of Rivals in Nest (反映 Nest 内部竞争强度)
for char in iv_chars:
    sum_nest = df.groupby(['market_ids', 'nesting_ids'])[char].transform('sum')
    df[f'iv_sum_nest_{char}'] = sum_nest - df[char]

# Differentiation (反映在这个 Nest 里是不是独一无二)
for char in iv_chars:
    df[f'iv_diff_nest_{char}'] = build_differentiation_ivs(
        df, char, ['market_ids', 'nesting_ids']
    )

for char in iv_chars:
    sum_nest = df.groupby(['market_ids', 'nesting_ids'])[char].transform('sum')
    df[f'iv_sum_nest_{char}'] = sum_nest - df[char]


print("\n   -> Pruning Weak IVs (Correlation Check)...")

# Gather all potential IVs
all_potential_ivs = [c for c in df.columns if c.startswith('iv_')]
iv_cols = all_potential_ivs
valid_ivs = []

dropped_ivs = []

# %%

print("-" * 50)
print(f"   -> Final Strong IV Set ({len(iv_cols)} variables):")
# Sort by strength for display
iv_strength = [(iv, df[[iv, 'price']].corr().iloc[0,1]) for iv in iv_cols]

for iv, corr in iv_strength:
    print(f"      {iv:<25} : {corr:.4f}")
print("-" * 50)

# %%
# ==============================================================================
# SECTION C: FINAL JOINT ESTIMATION (COMPLETE & STANDALONE)
# ==============================================================================
from scipy.optimize import minimize
import pyhdfe
import numpy as np
import pandas as pd

print("[3] Running FINAL JOINT Estimation (Soft Constrained)...")
print("   -> Strategy: Pooling groups + Interaction Terms + Soft Penalty")
df['log_ev_range_bev'] = df['log_ev_range'] * (df.fuel_type == 'BEV')
print("   -> Created 'log_ev_range_bev'. PHEV range effect set to 0.")
# ------------------------------------------------------------------------------
# 1. Configuration
# ------------------------------------------------------------------------------
TARGET_ALPHA = -3.0
LAMBDA_PENALTY = 500.0 
LAMBDA_PENALTY = 0.0
RANGE_PENALTY = 100.0  # Huge penalty if negative

# Shared Variables (Same for everyone) - INCLUDING both common and hetero
common_exog_vars = [
    'log_size',                    # Vehicle size/space
    'max_power',                   # Engine/Motor power (affects performance)
    # 'is_suv',                      # SUV body type dummy
    # 'is_mpv',                      # MPV body type dummy
    'log_ev_range_bev',            # EV range (BEV specific)
    # 'electric_efficiency',         # EV efficiency
    # 'fuel_consumption',            # Fuel/combustion efficiency (ICE/PHEV)
    'log_ev_power', 
    'log_battery_capacity',        # Battery size (EV specific)
    'log_net_price',               # Price coefficient (pooled across groups)
    'log_within_share'             # Nesting parameter (pooled across groups)
]

# 'max_power', 'log_size', 'log_ev_range', 'log_ev_power', 
#             'log_battery_capacity'

# Heterogeneous Variables (NOW EMPTY - all treated as homogeneous/pooled)
hetero_vars = []

# %%
# ------------------------------------------------------------------------------
# 2. Data Preparation & Interaction Terms
# ------------------------------------------------------------------------------
print("   -> Preparing Data & Interactions...")

# Filter Valid Data
df_joint = df[(df['shares'] > 0) ].copy()

# groups = sorted(df_joint['h'].unique())
# n_groups = len(groups)
n_common = len(common_exog_vars)



# A. Construct X Matrix (Regressors)
# -------------------------------------------------
# Structure: [Common | Price_G1, Rho_G1 | Price_G2, Rho_G2 | ...]
X_hetero_cols = []
# for grp in groups:
#     dummy = (df_joint['h'] == grp).astype(int)
#     for var in hetero_vars:
#         col_name = f"{var}_X_{grp}"
#         df_joint[col_name] = df_joint[var] * dummy
#         X_hetero_cols.append(col_name)

# B. Construct Z Matrix (Instruments)
# -------------------------------------------------
# Instruments must also be interacted to identify specific group parameters
Z_hetero_cols = []

# # dummy = (df_joint['h'] == grp).astype(int)
# for iv in iv_cols:
#     col_name = f"{iv}_X_{grp}"
#     df_joint[col_name] = df_joint[iv]
#     Z_hetero_cols.append(col_name)
Z_hetero_dm = df_joint[iv_cols].values
# %%
# ------------------------------------------------------------------------------
# 3. Absorb Fixed Effects (Residualization)
# ------------------------------------------------------------------------------
print("   -> Absorbing Brand x Group Fixed Effects...")

# Create Group-Specific Brand FE
# df_joint['fe_id'] = df_joint['brand'].astype(str) + "_" + df_joint['h'].astype(str)
absorb_fe = df_joint[['Brand', 'Year','Geo_Market']]
hdfe = pyhdfe.create(absorb_fe, drop_singletons=True)

def safe_resid(data):
    # Helper to residualize and clean NaNs
    res = hdfe.residualize(data)
    return np.nan_to_num(res) # Replace NaNs with 0 to prevent crashes

y_dm        = safe_resid(df_joint[['log_share_ratio']].values).flatten() # Flatten is crucial!
X_common_dm = safe_resid(df_joint[common_exog_vars].values)
if len(X_hetero_cols) > 0:
    X_hetero_dm = safe_resid(df_joint[X_hetero_cols].values)
    Z_hetero_dm = safe_resid(df_joint[Z_hetero_cols].values)
else:
    X_hetero_dm = np.empty((len(X_common_dm), 0))  # Empty array if no hetero vars
    Z_hetero_dm = np.empty((len(X_common_dm), 0))  # Empty array if no hetero vars - MUST match shape

# Ensure Z_hetero_dm is residualized properly
if Z_hetero_dm.shape[1] == 0:
    Z_hetero_dm = safe_resid(df_joint[iv_cols].values)

# Assemble Full Matrices
X_full = np.c_[X_common_dm, X_hetero_dm]
Z_full = np.c_[X_common_dm, Z_hetero_dm]

# Track variable names for later mapping
all_x_names = common_exog_vars + X_hetero_cols

# ------------------------------------------------------------------------------
# 4. Clean Dead Columns (CRITICAL STEP)
# ------------------------------------------------------------------------------
# Drop columns that have 0 variance (e.g., interaction terms for empty groups)
print("   -> Cleaning Matrices (Dropping Dead Columns)...")

def drop_zero_cols(mat, names):
    std = np.std(mat, axis=0)
    keep_mask = (std > 1e-9) & np.isfinite(std)
    if keep_mask.sum() < mat.shape[1]:
        print(f"      Dropped {mat.shape[1] - keep_mask.sum()} columns from {names}.")
    return mat[:, keep_mask], keep_mask

X_full_clean, X_keep_mask = drop_zero_cols(X_full, "X")
Z_full_clean, _ = drop_zero_cols(Z_full, "Z")

print(f"   -> Solver Input: N={len(y_dm)}, Vars={X_full_clean.shape[1]}")



# %%
# ------------------------------------------------------------------------------
print("[C.6] Running Optimization...")
n_groups = 1
# A. Point Estimates (Optimization)
betas_reduced = joint_soft_solver(
    y_dm, X_full_clean, Z_full_clean,
    n_common=n_common,
    target=TARGET_ALPHA,
    penalty=LAMBDA_PENALTY,
    keep_mask=X_keep_mask  # <--- NEW ARGUMENT
)
# B. Standard Errors (GMM Robust)
print("   -> Calculating GMM Robust Standard Errors...")

try:
    # 传入调整后的自由度 (Fixed Effects的数量)
    ses_reduced = calculate_gmm_se_robust(
        X_full_clean, Z_full_clean, y_dm, betas_reduced, 
        n_groups_to_adjust=n_groups
    )
except Exception as e:
    print(f"   [Warning] SE Calculation failed: {e}")
    ses_reduced = np.zeros_like(betas_reduced)

# %%
# --------------------------------------------------------------------------
# 7. Mapping & Display
# --------------------------------------------------------------------------
# Reconstruct full vectors (Point Est + SE)
betas_full = np.zeros(len(all_x_names))
ses_full = np.zeros(len(all_x_names))

# Map back using the mask (Active Columns Only)
betas_full[X_keep_mask] = betas_reduced
ses_full[X_keep_mask] = ses_reduced

print("\n" + "="*80)
print("FINAL ESTIMATION RESULTS (GMM ROBUST SE)")
print("="*80)

print("\n[SHARED COEFFICIENTS]")
print(f"{'Variable':<30} | {'Coef':>10} | {'SE':>10} | {'t-stat':>10} | {'Signif':<5}")
print("-" * 75)

for i, var in enumerate(common_exog_vars):
    b = betas_full[i]
    se = ses_full[i]
    
    if se > 1e-6:
        t_stat = b / se
        # Simple stars
        stars = ''
        if abs(t_stat) > 1.645: stars = '*'
        if abs(t_stat) > 1.96:  stars = '**'
        if abs(t_stat) > 2.58:  stars = '***'
    else:
        t_stat = np.nan
        stars = ''
        
    print(f"{var:<30} | {b:10.4f} | {se:10.4f} | {t_stat:10.2f} | {stars:<5}")

print("\n[GROUP-SPECIFIC PARAMETERS]")
print("-" * 95)
print(f"{'Group':<20} | {'Alpha':>9} {'(SE)':<8} | {'Rho':>9} {'(SE)':<8} | {'Elas Approx'}")
print("-" * 95)



# %%

# ------------------------------Not Adapted-----------------------------------
# ------------------------------------------------------------------------------
# 8. Store Results Back to DataFrame
results_rows = []
offset = n_common


# Save Results
df_results_final = pd.DataFrame(results_rows)
if 'alpha' in df.columns: df.drop(columns=['alpha', 'rho'], inplace=True)
df_est = df.merge(df_results_final, on='h', how='left')

# Fill shared betas
for i, var in enumerate(common_exog_vars):
    df_est[f'beta_{var}'] = betas_full[i]

print("-" * 75)
print("✅ Demand Estimation Done.")


# %%
# [5] Recalculate Xi (Absorb Residuals)
# ------------------------------------------------------------------------------
print("[5] Recalculating Xi (Unobserved Quality)...")

# 1. Calculate Delta (Mean Utility) from shares
# delta = log(s_j) - log(s_0) - rho * log(s_j|g)
delta = df_est['log_share_ratio'] - \
    (df_est['rho'] * df_est['log_within_share'])

# 2. Calculate Observable Part (Alpha*P + X*Beta)
val_xbeta = calculate_observable_utility(df_est)
val_price = df_est['alpha'] * df_est['log_net_price']

# 3. Xi = Delta - Observable
df_est['util_xi'] = delta - val_price - val_xbeta

print(
    f"   -> Xi Mean: {df_est['util_xi'].mean():.4f}, Std: {df_est['util_xi'].std():.4f}")

# %%
# [6] Validation
# ------------------------------------------------------------------------------
print("[6] Running Validation...")
df_est['util_base'] = calculate_total_utility(df_est)  # Should match Delta
df_est['share_pred'] = solve_nested_logit_shares(df_est, 'util_base')
err = (df_est['share_pred'] - df_est['shares']).abs().sum()
print(f"   -> Baseline Prediction Error: {err:.6e}")


print("-" * 80)
print("ROBUSTNESS CHECK: DEMAND BY YEAR")
print("-" * 80)


year_list = sorted(df_joint["year"].unique())
robustness_rows = []


# 例如：price coefficient 叫 alpha，特征系数是 beta_logevrange, beta_logsize
key_params = ["alpha", "beta_logevrange", "beta_logsize"]

for yr in year_list:
    print("-" * 60)
    print(f"Estimating demand using only year = {yr} ...")

    # 1. 子样本
    df_year = df_joint[df_joint["year"] == yr].copy()

    # 2. 构造 y, X, Z
    
    absorb_fe = df_year[['brand','h']]
    hdfe = pyhdfe.create(absorb_fe, drop_singletons=True)


    y_dm        = safe_resid(df_year[['log_share_ratio']].values).flatten() # Flatten is crucial!
    X_common_dm = safe_resid(df_year[common_exog_vars].values)
    X_hetero_dm = safe_resid(df_year[X_hetero_cols].values)
    Z_hetero_dm = safe_resid(df_year[Z_hetero_cols].values)

    # Assemble Full Matrices
    X_full = np.hstack([X_common_dm, X_hetero_dm])
    Z_full = np.hstack([X_common_dm, Z_hetero_dm])

    X_full_clean, X_keep_mask = drop_zero_cols(X_full, "X")
    Z_full_clean, _ = drop_zero_cols(Z_full, "Z")

    betas_year = joint_soft_solver(
        y_dm, X_full_clean, Z_full_clean,
        n_groups=n_groups,
        n_common=n_common,
        target=TARGET_ALPHA,
        penalty=LAMBDA_PENALTY,
        keep_mask=X_keep_mask  # <--- NEW ARGUMENT
    )

    # 4. Calculate SEs (GMM Robust)
    try:
        se_year = calculate_gmm_se_robust(
        X_full_clean, Z_full_clean, y_dm, betas_year, 
        n_groups_to_adjust=n_groups
    )
    except Exception as e:
        print(f"Year {yr}: SE calculation failed:", e)
        se_year = np.zeros_like(betas_year)

    # ----------
    # Reconstruct full vectors (Point Est + SE)
    betas_year_full = np.zeros(len(all_x_names))
    se_year_full = np.zeros(len(all_x_names))

    # Map back using the mask (Active Columns Only)
    betas_year_full[X_keep_mask] = betas_year
    se_year_full[X_keep_mask] = se_year

    print("\n" + "="*80)
    print(f"YEAR {yr} ESTIMATION RESULTS (GMM ROBUST SE)")
    print("="*80)

    print("\n[SHARED COEFFICIENTS]")
    print(f"{'Variable':<30} | {'Coef':>10} | {'SE':>10} | {'t-stat':>10} | {'Signif':<5}")
    print("-" * 75)

    for i, var in enumerate(common_exog_vars):
        b = betas_year_full[i]
        se = se_year_full[i]
        if se > 1e-6:
            t_stat = b / se
            stars = ''
            if abs(t_stat) > 1.645: stars = '*'
            if abs(t_stat) > 1.96:  stars = '**'
            if abs(t_stat) > 2.58:  stars = '***'
        else:
            t_stat = np.nan
            stars = ''
        print(f"{var:<30} | {b:10.4f} | {se:10.4f} | {t_stat:10.2f} | {stars:<5}")

    print("\n[GROUP-SPECIFIC PARAMETERS]")
    print("-" * 95)
    print(f"{'Group':<20} | {'Alpha':>9} {'(SE)':<8} | {'Rho':>9} {'(SE)':<8} | {'Elas Approx'}")
    print("-" * 95)

    offset = n_common
    for i, grp in enumerate(groups):
        full_idx_alpha = offset + i * 2
        full_idx_rho   = offset + i * 2 + 1
        alpha_val = betas_year_full[full_idx_alpha]
        alpha_se  = se_year_full[full_idx_alpha]
        rho_val   = betas_year_full[full_idx_rho]
        rho_se    = se_year_full[full_idx_rho]
        mean_p = 2.7
        elas = alpha_val * mean_p
        print(f"{grp:<20} | {alpha_val:9.4f} {f'({alpha_se:.3f})':<8} | {rho_val:9.4f} {f'({rho_se:.3f})':<8}")

    

# %%

# ==============================================================================
# SECTION D: SUPPLY & MC RECOVERY
# ==============================================================================

# [7] Supply Prep
# ------------------------------------------------------------------------------
print("[7] Preparing Supply Data & Recovering MC...")

agg_cols = {
    'net_price': 'first',
    'price': 'first',
    'subsidy': 'first',
    'brand': 'first',
    'nesting_ids': 'first',
    'max_power': 'first',
    'log_ev_range': 'first',
    'log_size': 'first',
    'fuel_consumption': 'first',
    'is_bev': 'first',
    'is_phev': 'first', # NEW
    'is_ice': 'first'   # NEW
}

df_supply = df_est.groupby(['Year', 'Models'], as_index=False).agg(agg_cols)
df_supply = df_supply.rename(columns={'net_price': 'net_price'})

all_brands = sorted(df_supply['brand'].unique())
brand_to_idx = {b: i for i, b in enumerate(all_brands)}
df_est['brand_idx'] = df_est['brand'].map(brand_to_idx)
df_supply['brand_idx'] = df_supply['brand'].map(brand_to_idx)

# Construct Base Utility
df_est['base_utility'] = df_est['util_xi'] + \
    calculate_observable_utility(df_est)
df_est['utility_hat'] = calculate_observable_utility(df_est)

# brand_fe = df_est.groupby(['brand', 'nesting_ids', 'h'])['util_xi'].transform('mean')
# df_est['utility_hat'] = df_est['utility_hat'] + brand_fe
# df_est['util_xi'] = df_est['util_xi'] - brand_fe

t_fe = df_est.groupby(['Year','brand','h'])['util_xi'].transform('mean')
df_est['utility_hat'] = df_est['utility_hat'] + t_fe
df_est['util_xi'] = df_est['util_xi'] - t_fe

print((df_est['base_utility'] - df_est['util_xi'] - df_est['utility_hat']).std())

# df_supply = df_supply.sort_values(['Year', 'brand_idx'])
df_supply['mc'] = 0.0

# %%
# [8] Recover MC
# ------------------------------------------------------------------------------
solver = FastEquilibriumSolver(df_est, df_supply)

mc_results = []
for t in df_supply['Year'].unique():
    mc_vec = solver.recover_mc(t)
    df_t = df_supply[df_supply['Year'] == t].copy().sort_values('Models')
    df_t['mc'] = mc_vec
    mc_results.append(df_t)

df_supply_clean = pd.concat(mc_results, ignore_index=True)

mask_bad = df_supply_clean['mc'] <= df_supply_clean['price'] * 0.1
if mask_bad.sum() > 0:
    print(f"   -> Fixing {mask_bad.sum()} negative MCs...")
    df_supply_clean.loc[mask_bad,
                        'mc'] = df_supply_clean.loc[mask_bad, 'price'] * 0.6

df_supply_clean['log_mc'] = np.log(df_supply_clean['mc'])

# %%

# 1. Construct Cost Drivers with Interactions
df_supply_clean['power_ice_phev'] = df_supply_clean['max_power'] * (1 - df_supply_clean['is_bev'])
df_supply_clean['size_ice_phev']  = df_supply_clean['log_size'] * (1 - df_supply_clean['is_bev'])
df_supply_clean['fuel_ice_phev']  = df_supply_clean['fuel_consumption'] # BEV 已经是 0 了

df_supply_clean['range_bev']      = df_supply_clean['log_ev_range'] * df_supply_clean['is_bev']
df_supply_clean['range_phev']     = df_supply_clean['log_ev_range'] * df_supply_clean['is_phev']

# 2. Define Regression Variables
reg_vars = [
    'power_ice_phev',  
    'size_ice_phev',   
    'fuel_ice_phev',   
    'range_bev', 
    'range_phev'
]

# 3. Prepare Data for Regression
df_reg = df_supply_clean.copy()
# 确保没有Inf/NaN
df_reg.replace([np.inf, -np.inf], np.nan, inplace=True)
df_reg.dropna(subset=reg_vars + ['log_mc'], inplace=True)

if len(df_reg) > 100:
    # 4. Residualize (Absorb Fixed Effects)
    print("   -> Absorbing FE for Supply Side...")

    hdfe = pyhdfe.create(df_reg[['Year', 'brand']], drop_singletons=True)
    
    y_r = hdfe.residualize(df_reg[['log_mc']].values)
    X_r = hdfe.residualize(df_reg[reg_vars].values)
    
    # 5. OLS Estimation (Point Estimates)
    # Beta = (X'X)^-1 X'y
    # Use pinv for numerical stability
    XtX_inv = np.linalg.pinv(X_r.T @ X_r)
    gamma = XtX_inv @ X_r.T @ y_r
    gamma = gamma.flatten()
    
    # 6. Calculate Robust Standard Errors (HC1)
    # Residuals
    residuals = y_r.flatten() - X_r @ gamma
    
    # Sandwich Variance Matrix: V = (X'X)^-1 * (X' diag(e^2) X) * (X'X)^-1
    # Meat part calculation (Optimized)
    u_sq = residuals**2
    # X' * diag(u^2) * X -> equivalent to (X.T * u_sq) @ X
    Meat = (X_r.T * u_sq) @ X_r
    
    V_robust = XtX_inv @ Meat @ XtX_inv
    
    # DOF Adjustment (N / N-K)
    N, K = X_r.shape
    dof_scale = N / max(1.0, N - K) 
    se_gamma = np.sqrt(np.diag(V_robust * dof_scale))
    
    # 7. Reconstruct MC_hat
    intercept = df_reg['log_mc'].mean() - np.dot(df_reg[reg_vars].mean().values, gamma)
    df_supply_clean['mc_hat'] = intercept + df_supply_clean[reg_vars].values @ gamma
    
    # 8. Results Display
    print("\n" + "="*60)
    print("SUPPLY SIDE RESULTS (Robust SE)")
    print("="*60)
    print(f"{'Cost Driver':<25} | {'Coef':>10} | {'SE':>10} | {'t-stat':>10} | {'Signif':<5}")
    print("-" * 68)
    
    coef_dict = dict(zip(reg_vars, gamma))
    
    for i, var in enumerate(reg_vars):
        b = gamma[i]
        se = se_gamma[i]
        t = b / se if se > 1e-9 else 0
        
        # Simple stars
        stars = ''
        if abs(t) > 1.645: stars = '*'
        if abs(t) > 1.96:  stars = '**'
        if abs(t) > 2.58:  stars = '***'
        
        print(f"{var:<25} | {b:10.4f} | {se:10.4f} | {t:10.2f} | {stars:<5}")

    print("-" * 68)
    
    # Economic Interpretation Logic
    is_size_pos = coef_dict.get('size_ice_phev', 0) > 0
    bev_slope = coef_dict.get('range_bev', 0)
    
    print(f"\n   -> Economic Checks:")
    print(f"      ICE Size Coeff:   {coef_dict.get('size_ice_phev', 0):.4f} "
          f"{'✅ (Positive)' if is_size_pos else '⚠️ (Negative - Check Model)'}")
    
    print(f"      BEV Range Slope:  {bev_slope:.4f} "
          f"{'✅ (Sensitive)' if bev_slope > 0.15 else '⚠️ (Low Sensitivity)'}")

    # Omega
    df_supply_clean['cost_omega'] = df_supply_clean['log_mc'] - df_supply_clean['mc_hat']
    
else:
    print("⚠️ Not enough data for Supply Side Estimation.")
    df_supply_clean['mc_hat'] = df_supply_clean['log_mc']

# Final Merge
df_final_sup = df_supply_clean.copy()


# brand_fe = df_final_sup.groupby(['brand', 'nesting_ids'])['cost_omega'].transform('mean')
# df_final_sup['mc_hat'] = df_final_sup['mc_hat'] + brand_fe
# df_final_sup['cost_omega'] = df_final_sup['cost_omega'] - brand_fe

t_fe = df_supply_clean.groupby(['Year', 'brand', 'nesting_ids'])['cost_omega'].transform('mean')
df_supply_clean['mc_hat'] = df_supply_clean['mc_hat'] + t_fe
df_supply_clean['cost_omega'] = df_supply_clean['cost_omega'] - t_fe

print((df_supply_clean['mc'].apply(np.log) - df_supply_clean['cost_omega'] - df_supply_clean['mc_hat']).std())

# %%
# ==============================================================================
# SECTION E: EXPORT
# ==============================================================================
print("[E] Saving Results...")
out_dir = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(out_dir, exist_ok=True)

# Select Demand Columns
dem_cols = ['Year', 'nesting_ids', 'h', 'Models', 'h_weight', 'brand', 'shares', 'price', 'net_price', 'alpha', 'rho', 'subsidy',
            'util_xi', 'util_base', 'base_utility', 'utility_hat',
            'is_bev', 'is_phev', 'max_power', 'log_ev_range', 'market_size'] 
dem_cols += [c for c in df_est.columns if c.startswith('beta_')]
df_est[list(set(dem_cols))].to_csv(os.path.join(
    out_dir, "demand_results.csv"), index=False)

# Supply Columns
sup_cols = ['Year', 'brand', 'price', 'mc', 'Models', 'log_mc', 'mc_hat', 'cost_omega',
            'is_bev', 'is_phev', 'max_power', 'log_ev_range', 'subsidy', 'nesting_ids']
df_final_sup[list(set(sup_cols))].to_csv(
    os.path.join(out_dir, "supply_results.csv"), index=False)

print("Done.")
# %%
# ==============================================================================
# SECTION 9b: TABULATION OF RESULTS (BY HOUSEHOLD GROUP)
# ==============================================================================
print("\n" + "="*80)
print("FINAL ESTIMATION RESULTS - HETEROGENEITY ANALYSIS")
print("="*80)

# Identify all demand parameters
demand_params = ['alpha', 'rho'] + [c for c in df_est.columns if c.startswith('beta_')]

# Get unique household groups
unique_groups = sorted(df_est['h'].unique())

# --- 1. Demand Side Tabulation (Loop per 'h') ---
for h_val in unique_groups:
    df_sub = df_est[df_est['h'] == h_val]
    
    # Skip if for some reason this group has no rows (though unlikely given previous steps)
    if df_sub.empty:
        continue

    print(f"\n[Household Group: {h_val}] (N = {len(df_sub)})")
    print("-" * 65)
    print(f"{'Parameter':<30} | {'Mean':>10} | {'Std Dev':>10} | {'Min':>8}")
    print("-" * 65)

    # Calculate summary stats for this specific group
    dem_stats = df_sub[demand_params].describe().T

    for param, row in dem_stats.iterrows():
        # Clean up name for display
        clean_name = param.replace("beta_", "").replace("log_", "ln(").replace("_", " ").title()
        if param == 'alpha': clean_name = 'Price Sensitivity (Alpha)'
        if param == 'rho': clean_name = 'Nesting Param (Rho)'
        
        print(f"{clean_name:<30} | {row['mean']:10.4f} | {row['std']:10.4f} | {row['min']:8.4f}")
    
    print("-" * 65)

print(f"\nTotal Households Groups Tabulated: {len(unique_groups)}")


# --- 2. Supply Side Tabulation (Global) ---
# Supply function parameters are typically estimated globally (pooled), 
# so we keep this as a single table unless you ran supply regressions by h.
print("\n\n" + "="*80)
print("[Supply Side Parameters (Marginal Cost Function - Global)]")
print("-" * 65)
print(f"{'Cost Driver':<30} | {'Coeff':>10} | {'Interpretation':<20}")
print("-" * 65)

var_map = {
    'power_ice_phev': 'HP Cost (ICE/PHEV)',
    'size_ice_phev':  'Size Cost (ICE/PHEV)',
    'fuel_ice_phev':  'Eff. Tech Cost (ICE)',
    'range_bev':      'Battery Cost (BEV)',
    'range_phev':     'Battery Cost (PHEV)'
}

if 'gamma' in locals():
    for var_name, coeff in zip(reg_vars, gamma):
        readable = var_map.get(var_name, var_name)
        
        # logical check for interpretation column
        interp = ""
        if "range" in var_name:
            interp = "High" if coeff > 0.5 else ("Low" if coeff < 0.1 else "Reasonable")
        elif "power" in var_name or "size" in var_name:
            interp = "Expected (+)" if coeff > 0 else "Unexpected (-)"
            
        print(f"{readable:<30} | {coeff:10.4f} | {interp:<20}")
else:
    print("Supply estimation was skipped or gamma not found.")
print("-" * 65)
print("\n")
# %%
