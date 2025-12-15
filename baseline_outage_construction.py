import numpy as np
import pandas as pd


def baseline_outage_construction(storms_data, outage_df):
    storms = storms_data[["CZ_FIPS", "BEGIN_DATE_TIME", "END_DATE_TIME"]].copy()
    storms["BEGIN_DATE_TIME"] = pd.to_datetime(storms["BEGIN_DATE_TIME"])
    storms["END_DATE_TIME"] = pd.to_datetime(storms["END_DATE_TIME"])
    storms["CZ_FIPS"] = storms["CZ_FIPS"].astype(int)
    
    outage = outage_df[["fips_code", "run_start_time", "sum"]].copy()
    outage = outage.rename(columns={"fips_code": "CZ_FIPS"})
    outage["CZ_FIPS"] = outage["CZ_FIPS"].astype(int)
    outage["run_start_time"] = pd.to_datetime(outage["run_start_time"])
    
    
    def compute_baseline_for_county(df_out, df_storm):
        if df_storm.empty:
            return df_out["sum"].median()
    
        t = df_out["run_start_time"].values
        is_storm = np.zeros(len(df_out), dtype=bool)
    
        for _, r in df_storm.iterrows():
            is_storm |= (t >= r["BEGIN_DATE_TIME"]) & (t <= r["END_DATE_TIME"])
    
        baseline_vals = df_out.loc[~is_storm, "sum"]
    
        if baseline_vals.empty:
            return np.nan
    
        return baseline_vals.median()
    
    baseline_records = []
    
    for cz, df_out_c in outage.groupby("CZ_FIPS"):
        df_storm_c = storms.loc[storms["CZ_FIPS"] == cz]
        baseline = compute_baseline_for_county(df_out_c, df_storm_c)
    
        baseline_records.append(
            {
                "CZ_FIPS": cz,
                "baseline_outage_median": baseline,
            }
        )
    baseline_df = pd.DataFrame(baseline_records)
    baseline_df = baseline_df.copy()
    baseline_df.columns = baseline_df.columns.str.strip()
    
    storms_data = storms_data.copy()
    storms_data.columns = storms_data.columns.str.strip()
    
    baseline_df["fips5"] = baseline_df["CZ_FIPS"].astype(str).str.zfill(5)
    
    candidate_cols = [c for c in baseline_df.columns if c not in {"CZ_FIPS", "fips5"}]

    
    preferred = [c for c in candidate_cols if "baseline" in c.lower() or "median" in c.lower()]
    baseline_col = preferred[0] if preferred else candidate_cols[0]

    
    if "full_fips" in storms_data.columns:
        storms_data["fips5"] = pd.to_numeric(storms_data["full_fips"], errors="coerce").astype("Int64").astype(str).str.zfill(5)
    elif "fips_str" in storms_data.columns:
        storms_data["fips5"] = storms_data["fips_str"].astype(str).str.extract(r"(\d+)")[0].str.zfill(5)
    else:
        raise KeyError("no full_fips or fips_str")

    tmp = baseline_df[["fips5", baseline_col]].rename(columns={baseline_col: "baseline_outage_median"})

    if "baseline_outage_median" in storms_data.columns:
        storms_data = storms_data.drop(columns=["baseline_outage_median"])
    
    storms_data = storms_data.merge(
        tmp,
        on="fips5",
        how="left",
        validate="many_to_one",
    )
    return storms_data