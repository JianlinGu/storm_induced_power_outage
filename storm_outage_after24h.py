import pandas as pd

def match_max_outage_after24h(
    storm_df: pd.DataFrame,
    outage_df: pd.DataFrame,
    storm_time_col: str = "BEGIN_DATE_TIME",
    storm_state_col: str = "STATE",
    storm_county_col: str = "CZ_NAME",
    outage_time_col: str = "run_start_time",
    outage_state_col: str = "state",
    outage_county_col: str = "county",
    outage_value_col: str = "sum",
    result_col: str = "max_outage_after_24h",
    verbose: bool = False,
) -> pd.DataFrame:

    storm_df = storm_df.copy()
    outage_df = outage_df.copy()
    storm_df[storm_time_col] = pd.to_datetime(storm_df[storm_time_col], errors="coerce")
    outage_df[outage_time_col] = pd.to_datetime(outage_df[outage_time_col], errors="coerce")
    storm_df["_county_clean"] = storm_df[storm_county_col].astype(str).str.strip().str.upper()
    storm_df["_state_clean"] = storm_df[storm_state_col].astype(str).str.strip().str.upper()
    outage_df["_county_clean"] = outage_df[outage_county_col].astype(str).str.strip().str.upper()
    outage_df["_state_clean"] = outage_df[outage_state_col].astype(str).str.strip().str.upper()

    storm_df["_loc_key"] = storm_df["_state_clean"] + "||" + storm_df["_county_clean"]
    outage_df["_loc_key"] = outage_df["_state_clean"] + "||" + outage_df["_county_clean"]

    outage_slim = (
        outage_df[["_loc_key", outage_time_col, outage_value_col]]
        .dropna(subset=[outage_time_col])
        .sort_values(["_loc_key", outage_time_col])
    )

    storm_valid = storm_df[storm_df[storm_time_col].notna()].copy()
    storm_valid["_storm_idx"] = storm_valid.index
    storm_valid["_after_24h_end"] = storm_valid[storm_time_col] + pd.Timedelta(hours=24)

    results = []

    for loc in storm_valid["_loc_key"].unique():
        storm_loc = storm_valid[storm_valid["_loc_key"] == loc]
        outage_loc = outage_slim[outage_slim["_loc_key"] == loc]

        if outage_loc.empty:
            for idx in storm_loc["_storm_idx"]:
                results.append({"_storm_idx": idx, result_col: 0})
            continue

        for _, srow in storm_loc.iterrows():
            mask = (
                (outage_loc[outage_time_col] >= srow[storm_time_col]) &
                (outage_loc[outage_time_col] <= srow["_after_24h_end"])
            )
            max_outage = outage_loc.loc[mask, outage_value_col].max() if mask.any() else 0
            results.append({"_storm_idx": srow["_storm_idx"], result_col: max_outage})

    res_df = pd.DataFrame(results)
    storm_df[result_col] = 0
    storm_df.loc[res_df["_storm_idx"], result_col] = res_df[result_col].values

    # --- 清理 ---
    storm_df.drop(
        columns=["_loc_key", "_county_clean", "_state_clean"],
        inplace=True,
        errors="ignore",
    )

    return storm_df