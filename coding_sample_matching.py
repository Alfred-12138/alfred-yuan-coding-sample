# Main Matching Function

# Code description: This code contains the main matching function. 
# The matching logic is as follows: For each row in the sales data, 
# the search first takes place within the subset of the price data 
# where fuel type and displacement are the same. 
# The search method scores the similarity of the brand, model, and trim strings. 
# If the model is highly similar, or if the brand is similar 
# and the total score is relatively high, the match is considered successful. 
# If the match fails, the constraints on fuel type and displacement are relaxed, 
# and the search is repeated in the full table. If this also fails, 
# the row is classified as unmatched.

import warnings
from collections import Counter
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

# =====================================================
# Column configuration: adapt these to your real data.
# The Chinese column names are project-specific.
# =====================================================

PRICE_COLS = {
    "trim": "车型名称",          # trim name in the price table
    "model": "车系名称",         # model/series name in the price table
    "brand": "品牌名称",         # brand name in the price table
    "fuel": "能源类型",          # fuel/energy type
    "year": "年份",             # year
    "body": "车身结构",          # body type (SUV / sedan / etc.)
    "disp": "排量(L)",           # displacement (liters)
    "index": "car_index",        # unique car identifier defined by you
    "msrp": "厂商指导价(元)",     # manufacturer suggested retail price
}

SALES_COLS = {
    "trim": "款型",             # trim name in the sales table
    "model": "车型",            # model/series name in the sales table
    "brand": "品牌",            # brand name in the sales table
    "fuel": "燃料",             # fuel type in the sales table
    "year": "年份(提取)",       # extracted year from the sales data
    "body": "车身形式",         # body type
    "disp": "排量(仅数字)",      # numeric displacement
}


# =====================================================
# String similarity: Sørensen–Dice coefficient
# =====================================================

def match_score(a: str, b: str) -> float:
    """
    Compute Sørensen–Dice similarity between two strings in [0, 1].

    Intuition
    ---------
    - Treat the two strings as multisets of characters.
    - Similarity = 2 * |intersection| / (len(a) + len(b)).

    Returns NaN if either string is missing or empty.
    """
    if pd.isna(a) or pd.isna(b):
        return np.nan

    a, b = str(a).strip(), str(b).strip()
    if not a or not b:
        return np.nan

    A, B = Counter(a), Counter(b)
    inter = sum((A & B).values())
    return 2 * inter / (len(a) + len(b))


# =====================================================
# Year penalty: prefer closer years
# =====================================================

def year_penalty(query_year: float, candidate_year: float) -> float:
    """
    Negative absolute difference in years.

    Intuition
    ---------
    - The closer the candidate year is to the query year, the better.
    - We penalize large gaps by subtracting |candidate - query|.
    - If either year is missing, return 0 (no penalty / bonus).
    """
    if pd.isna(query_year) or pd.isna(candidate_year):
        return 0.0
    return -abs(candidate_year - query_year)


# =====================================================
# Main matching function
# =====================================================

def data_matching(
    df_price: pd.DataFrame,
    df_sales: pd.DataFrame,
    threshold_if_trim: float = 40.0,
    max_diff: int = 2,
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Match car sales records to price records at the trim level.

    This function implements a multi-step matching procedure:
    1. Use (brand, model, fuel) to build a candidate subset in the price table.
    2. If possible, further restrict by exact displacement and body type.
    3. Apply a year window (|year_price - year_sales| <= max_diff).
    4. Within the subset, compute:
       - trim similarity: match_score(trim_sales, trim_price)
       - year penalty:   year_penalty(year_sales, year_price)
       and define total score = 100 * trim_score + 10 * year_penalty.
    5. If no good match is found, relax conditions and fall back to
       a looser scheme.

    Parameters
    ----------
    df_price : pd.DataFrame
        Price dataset. Must contain columns specified in PRICE_COLS.
    df_sales : pd.DataFrame
        Sales dataset. Must contain columns specified in SALES_COLS.
    threshold_if_trim : float, optional
        Minimal total score required when a trim string is available.
        If the trim is relatively informative, we require a higher standard.
    max_diff : int, optional
        Maximum allowed year difference in the strictest stage.
    show_progress : bool, optional
        Whether to show a tqdm progress bar during matching.

    Returns
    -------
    pd.DataFrame
        The original sales DataFrame, augmented with:
        - matched price information (car_index, MSRP, etc.);
        - matching scores (score_trim, score_year_pen, total_score).
    """

    # Short aliases for frequently used column names
    trim_price = PRICE_COLS["trim"]
    model_price = PRICE_COLS["model"]
    brand_price = PRICE_COLS["brand"]

    trim_sales = SALES_COLS["trim"]
    model_sales = SALES_COLS["model"]
    brand_sales = SALES_COLS["brand"]
    fuel_sales = SALES_COLS["fuel"]
    disp_sales = SALES_COLS["disp"]
    year_sales = SALES_COLS["year"]
    bodytype_sales = SALES_COLS["body"]

    # Columns from the price table that we want to keep in the final result
    price_match_col = [brand_price, model_price, trim_price, PRICE_COLS["fuel"]]
    price_verify_col = [PRICE_COLS["year"], PRICE_COLS["body"], PRICE_COLS["disp"]]
    price_info_col = [PRICE_COLS["index"], PRICE_COLS["msrp"]]
    df1_keep = price_match_col + price_verify_col + price_info_col

    # Work on a reduced version of the price table
    df1_use = df_price[df1_keep].copy()

    # Keys for matching: we treat (brand, model, trim, fuel) as a group
    keys = [brand_sales, model_sales, trim_sales, fuel_sales]

    # Reduced version of the sales table (only relevant columns)
    df2_use = df_sales[
        [trim_sales, model_sales, brand_sales, fuel_sales, disp_sales, year_sales, bodytype_sales]
    ].copy()

    # Unique combinations of (brand, model, trim, fuel) in the sales data
    uniq = df2_use.drop_duplicates(subset=keys, keep="first").reset_index(drop=True)

    # -----------------------------------------------------
    # Internal helper: subset by brand + model (+ fuel)
    # -----------------------------------------------------
    def _subset_exact_bmf(
        df: pd.DataFrame, brand: str, model: str, fuel: str
    ) -> pd.DataFrame:
        """
        Step 1: strict subset by (brand, model, fuel).

        Logic
        -----
        - Brand and model must match exactly.
        - Fuel can be slightly relaxed: if fuel is missing or blank in the
          price table, we still keep that row as a candidate.
        """
        if df.empty:
            return df.iloc[0:0]

        b = "" if pd.isna(brand) else str(brand).strip()
        m = "" if pd.isna(model) else str(model).strip()
        f = None if pd.isna(fuel) or str(fuel).strip() == "" else str(fuel).strip()

        if b == "" or m == "":
            # If brand or model is missing, we cannot build a reliable subset
            return df.iloc[0:0]

        cond_brand = df[brand_price] == b
        cond_model = df[model_price] == m

        if f is None:
            cond_fuel = True
        else:
            cond_fuel = (
                df[PRICE_COLS["fuel"]].astype(str).str.strip().eq(f)
                | df[PRICE_COLS["fuel"]].isna()
                | (df[PRICE_COLS["fuel"]].astype(str).str.strip() == "")
            )

        return df[cond_brand & cond_model & cond_fuel]

    # -----------------------------------------------------
    # Internal helper: further subset by displacement + body
    # -----------------------------------------------------
    def _subset_add_disp_body(
        df: pd.DataFrame, displacement: float, body_type: str
    ) -> pd.DataFrame:
        """
        Step 2: refine the subset by requiring exact displacement and body type,
        if both pieces of information are available.

        If displacement or body type is missing, an empty DataFrame is returned,
        which will trigger fall-back logic later.
        """
        if df.empty:
            return df

        if pd.isna(displacement) or pd.isna(body_type):
            return df.iloc[0:0]

        bt = str(body_type).strip()
        return df[(df[PRICE_COLS["disp"]] == displacement) & (df[PRICE_COLS["body"]] == bt)]

    # -----------------------------------------------------
    # Internal helper: year window filter
    # -----------------------------------------------------
    def _apply_year_window(df: pd.DataFrame, q_year: float, window: int) -> pd.DataFrame:
        """
        Restrict the candidate set to rows within a given year window.
        """
        if df.empty or pd.isna(q_year):
            return df
        return df[(df[PRICE_COLS["year"]] - q_year).abs() <= window]

    # -----------------------------------------------------
    # Internal helper: pick the best candidate within a subset
    # -----------------------------------------------------
    def _pick_by_trim_year(
        sub: pd.DataFrame, q_trim: str, q_year: float
    ) -> Optional[Tuple[pd.Series, float, float, float]]:
        """
        Given a candidate subset, compute trim similarity and year penalty,
        then select the row with the highest total score.

        Returns
        -------
        (row, trim_score, year_score, total_score) or None
        """
        if sub is None or sub.empty:
            return None

        trim_scores = sub[trim_price].apply(lambda x: match_score(q_trim, x)).to_numpy(float)
        year_scores = sub[PRICE_COLS["year"]].apply(
            lambda x: year_penalty(q_year, x)
        ).to_numpy(float)

        # Total score: simple linear combination
        totals = 100 * np.nan_to_num(trim_scores, nan=0.0) + 10 * np.nan_to_num(
            year_scores, nan=0.0
        )

        # If the trim string seems informative (non-empty and not containing
        # special patterns like "出租"), we require the total score to exceed
        # a threshold to accept the match.
        has_trim = bool(q_trim) and (not pd.isna(q_trim)) and ("出租" not in str(q_trim))
        if has_trim:
            mask = np.nan_to_num(totals, nan=-np.inf) >= float(threshold_if_trim)
            if not mask.any():
                return None

            totals_masked = np.where(mask, totals, -np.inf)
            k = int(np.nanargmax(totals_masked))
            if not np.isfinite(totals_masked[k]):
                return None

            row = sub.iloc[k]
            return row, trim_scores[k], year_scores[k], totals[k]

        # If the trim information is not reliable, simply pick the maximum
        k = int(np.nanargmax(np.nan_to_num(totals, nan=-np.inf)))
        row = sub.iloc[k]
        return row, trim_scores[k], year_scores[k], totals[k]

    # =====================================================
    # Main loop: iterate over unique (brand, model, trim, fuel)
    # =====================================================

    out_car_index, out_score_trim, out_score_year, out_total_score = [], [], [], []

    iterator = uniq.iterrows()
    if show_progress:
        iterator = tqdm(iterator, total=len(uniq), desc="Matching", unit="row")

    for _, r in iterator:
        q_brand = r[brand_sales]
        q_model = r[model_sales]
        q_trim = r[trim_sales]
        q_fuel = r[fuel_sales]
        q_disp = r.get(disp_sales, np.nan)
        q_year = r.get(year_sales, np.nan)
        q_body = r.get(bodytype_sales, np.nan)

        # Stage 1: strict subset with brand+model+fuel,
        #          then displacement+body, then narrow year window.
        bmf = _subset_exact_bmf(df1_use, q_brand, q_model, q_fuel)
        bmf_strict = _subset_add_disp_body(bmf, q_disp, q_body)
        sub1 = _apply_year_window(bmf_strict, q_year, max_diff)
        picked = _pick_by_trim_year(sub1, q_trim, q_year)

        # Stage 2: if Stage 1 fails, relax displacement/body,
        #          only require brand+model+fuel + wider year window.
        if picked is None:
            sub2 = _apply_year_window(bmf, q_year, window=5)
            picked = _pick_by_trim_year(sub2, q_trim, q_year)

        # If we still cannot find a match, record NaNs
        if picked is None:
            out_car_index.append(np.nan)
            out_score_trim.append(np.nan)
            out_score_year.append(np.nan)
            out_total_score.append(np.nan)
            continue

        # Otherwise, record the chosen candidate and scores
        chosen, t_s, y_s, tot = picked
        out_car_index.append(chosen[PRICE_COLS["index"]])
        out_score_trim.append(t_s)
        out_score_year.append(y_s)
        out_total_score.append(tot)

    # Build a mapping table at the unique (brand, model, trim, fuel) level
    mapping = uniq.copy()
    mapping["car_index"] = out_car_index
    mapping["score_trim"] = out_score_trim
    mapping["score_year_pen"] = out_score_year
    mapping["total_score"] = out_total_score

    # Attach price information via car_index
    mapping = mapping.merge(
        df_price[df1_keep],
        left_on="car_index",
        right_on=PRICE_COLS["index"],
        how="left",
    )

    # Basic summary statistics for the matching quality
    print("Number of unique (brand, model, trim, fuel) combinations:", len(mapping))
    print("Number of unmatched rows:", mapping["total_score"].isna().sum())
    print("Share unmatched (%):", mapping["total_score"].isna().mean() * 100)

    # Drop columns from the sales side that would duplicate columns in mapping
    mapping = mapping.drop(
        columns=[SALES_COLS["disp"], SALES_COLS["year"], SALES_COLS["body"]],
        errors="ignore",
    )

    # Merge the mapping back into the full sales table
    result = df_sales.merge(mapping, on=keys, how="left")

    return result



# Preserve the existing capability for independent execution
if __name__ == "__main__":

    import pandas as pd
    # Creat Test Data
    test_df1 = pd.DataFrame({...})
    test_df2 = pd.DataFrame({...})
    mapping, not_matched = data_matching(test_df1, test_df2)