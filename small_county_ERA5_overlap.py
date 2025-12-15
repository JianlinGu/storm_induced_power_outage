import re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

def small_county_ERA5_overlap(ERA5_DIR, storms_data):
    BAD_FIPS = {"34023","25025","34017","34039","44003","36047","44001","36005"}
    FIPS_COL = "full_fips"
    LAT_COL_STORM = "LATITUDE"
    LON_COL_STORM = "LONGITUDE"
    STORM_TIME_COL = "BEGIN_DATE_TIME"
    
    # ERA5 columns
    TIME_COL = "valid_time"
    LAT_COL_ERA = "latitude"
    LON_COL_ERA = "longitude"
    ERA_COLS = ["tp", "i10fg", "crr"]
    
    OUT_I10FG = "era_i10fg_max_total_48h"
    OUT_TP    = "era_tp_max_total_48h"
    OUT_CRR   = "era_crr_max_total_48h"
    
    HOURS_HALF = 24          
    CHUNKSIZE  = 1_200_000   
    BBOX_PAD_DEG = 1.0      
    

    need = [FIPS_COL, LAT_COL_STORM, LON_COL_STORM, STORM_TIME_COL]
    miss = [c for c in need if c not in storms_data.columns]
    if miss:
        raise KeyError(f"storms_data missing columns: {miss}")
    
    storms_data[FIPS_COL] = storms_data[FIPS_COL].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(5)
    storms_data[STORM_TIME_COL] = pd.to_datetime(storms_data[STORM_TIME_COL], errors="coerce")
    if storms_data[STORM_TIME_COL].isna().any():
        raise ValueError(f"{STORM_TIME_COL} has NaT. Fix parsing first.")
    
    for c in [OUT_I10FG, OUT_TP, OUT_CRR]:
        if c not in storms_data.columns:
            storms_data[c] = np.nan
    
    mask_bad = storms_data[FIPS_COL].isin(BAD_FIPS)
    storm_bad = storms_data.loc[mask_bad].copy()
    if storm_bad.empty:
        print("No BAD_FIPS rows. Nothing to do.")
        raise SystemExit(0)
    
    # bbox over BAD_FIPS storms
    lat_min = float(storm_bad[LAT_COL_STORM].min()) - BBOX_PAD_DEG
    lat_max = float(storm_bad[LAT_COL_STORM].max()) + BBOX_PAD_DEG
    lon_min = float(storm_bad[LON_COL_STORM].min()) - BBOX_PAD_DEG
    lon_max = float(storm_bad[LON_COL_STORM].max()) + BBOX_PAD_DEG
    
    storm_bad["t0"] = storm_bad[STORM_TIME_COL].dt.floor("h")
    
    storm_bad["t_start"] = storm_bad["t0"] - pd.to_timedelta(HOURS_HALF, unit="h")
    storm_bad["t_end"]   = storm_bad["t0"] + pd.to_timedelta(HOURS_HALF, unit="h")

    years_needed = sorted(set(
        pd.concat([storm_bad["t_start"], storm_bad["t_end"]]).dt.year.unique().tolist()
    ))
    
    required_times_by_year = {}
    for y in years_needed:
        # collect all window hours that fall in this year
        times = []
        sub = storm_bad[(storm_bad["t_start"].dt.year <= y) & (storm_bad["t_end"].dt.year >= y)]
        for _, r in sub.iterrows():
            # clamp to year range
            a = r["t_start"]
            b = r["t_end"]
            a2 = max(a, pd.Timestamp(f"{y}-01-01 00:00:00"))
            b2 = min(b, pd.Timestamp(f"{y}-12-31 23:00:00"))
            if a2 <= b2:
                times.append(pd.date_range(a2, b2, freq="h"))
        if times:
            all_times = pd.DatetimeIndex(np.unique(np.concatenate([t.values for t in times])))
            required_times_by_year[y] = set(all_times.values)  # set of numpy datetime64[ns]
        else:
            required_times_by_year[y] = set()
    
    print("Years needed:", years_needed)
    
    YEAR_RE = re.compile(r"data\s*(\d{4}).*\.csv$", re.IGNORECASE)
    
    def find_year_files(era_dir: Path) -> dict:
        out = {}
        for p in sorted(era_dir.glob("data*.csv")):
            m = YEAR_RE.search(p.name)
            if m:
                out[int(m.group(1))] = p
        if not out:
            raise FileNotFoundError(f"No ERA5 yearly CSVs found in {era_dir} matching 'data*YYYY*.csv'")
        return out
    
    YEAR2PATH = find_year_files(ERA5_DIR)

    sample_year = next((y for y in years_needed if y in YEAR2PATH), None)
    sample_path = YEAR2PATH[sample_year]

    grid_points = []
    usecols_grid = [LAT_COL_ERA, LON_COL_ERA]
    for chunk in pd.read_csv(sample_path, usecols=usecols_grid, chunksize=CHUNKSIZE):
        m = (
            (chunk[LAT_COL_ERA] >= lat_min) & (chunk[LAT_COL_ERA] <= lat_max) &
            (chunk[LON_COL_ERA] >= lon_min) & (chunk[LON_COL_ERA] <= lon_max)
        )
        chunk = chunk.loc[m]
        if chunk.empty:
            continue
        grid_points.append(chunk.drop_duplicates())
    
    grid_df = pd.concat(grid_points, ignore_index=True).drop_duplicates()

    
    GRID = grid_df[[LAT_COL_ERA, LON_COL_ERA]].to_numpy()
    TREE = BallTree(np.radians(GRID), metric="haversine")
    
    def nearest_grid(lat, lon):
        q = np.radians(np.array([[float(lat), float(lon)]]))
        _, idx = TREE.query(q, k=1)
        i = int(idx[0, 0])
        return float(GRID[i, 0]), float(GRID[i, 1])
    
    storm_bad["grid_latlon"] = storm_bad.apply(lambda r: nearest_grid(r[LAT_COL_STORM], r[LON_COL_STORM]), axis=1)
    storm_bad["grid_lat"] = storm_bad["grid_latlon"].apply(lambda x: x[0])
    storm_bad["grid_lon"] = storm_bad["grid_latlon"].apply(lambda x: x[1])
    
    def read_era_year_filtered(year: int) -> pd.DataFrame:
        """Read one year's ERA5 CSV, but keep only:
           - bbox rows
           - rows whose valid_time is in required_times_by_year[year]
        """
        if year not in YEAR2PATH:
            return pd.DataFrame(columns=[TIME_COL, LAT_COL_ERA, LON_COL_ERA] + ERA_COLS)
    
        timeset = required_times_by_year.get(year, set())
        if not timeset:
            return pd.DataFrame(columns=[TIME_COL, LAT_COL_ERA, LON_COL_ERA] + ERA_COLS)
    
        path = YEAR2PATH[year]
        usecols = [TIME_COL, LAT_COL_ERA, LON_COL_ERA] + ERA_COLS
        kept = []
    
        for chunk in pd.read_csv(path, usecols=usecols, chunksize=CHUNKSIZE):
            # bbox first (cheap)
            m = (
                (chunk[LAT_COL_ERA] >= lat_min) & (chunk[LAT_COL_ERA] <= lat_max) &
                (chunk[LON_COL_ERA] >= lon_min) & (chunk[LON_COL_ERA] <= lon_max)
            )
            chunk = chunk.loc[m].copy()
            if chunk.empty:
                continue
    
            # parse valid_time
            chunk[TIME_COL] = pd.to_datetime(chunk[TIME_COL], errors="coerce").dt.floor("h")
            chunk = chunk.dropna(subset=[TIME_COL])
    
            # time filter
            # Convert to numpy datetime64[ns] for set membership
            tvals = chunk[TIME_COL].values
            mask_t = np.isin(tvals, list(timeset))
            chunk = chunk.loc[mask_t].copy()
            if chunk.empty:
                continue
    
            # numeric
            for c in ERA_COLS:
                chunk[c] = pd.to_numeric(chunk[c], errors="coerce")
    
            kept.append(chunk)
    
        if not kept:
            return pd.DataFrame(columns=[TIME_COL, LAT_COL_ERA, LON_COL_ERA] + ERA_COLS)
    
        df = pd.concat(kept, ignore_index=True)
        return df
    
    # Load filtered ERA data per year (only the exact hours you need)
    era_by_year = {}
    for y in years_needed:

        era_by_year[y] = read_era_year_filtered(y)

    
    # Concatenate for easy querying across boundary years
    era_all = pd.concat([df for df in era_by_year.values() if not df.empty], ignore_index=True)
    if era_all.empty:
        raise RuntimeError("No ERA rows matched your required times. Check TIME_COL parsing and storm times.")

    era_all[TIME_COL] = pd.to_datetime(era_all[TIME_COL], errors="coerce").dt.floor("H")

    def agg_one(row):
        t_start = row["t_start"]
        t_end   = row["t_end"]
        glat    = row["grid_lat"]
        glon    = row["grid_lon"]
    
        sub = era_all[
            (era_all[LAT_COL_ERA] == glat) &
            (era_all[LON_COL_ERA] == glon) &
            (era_all[TIME_COL] >= t_start) &
            (era_all[TIME_COL] <= t_end)
        ]
        if sub.empty:
            return pd.Series([np.nan, np.nan, np.nan], index=[OUT_I10FG, OUT_TP, OUT_CRR])
    
        i10fg_val = sub["i10fg"].max(skipna=True)
        tp_val    = sub["tp"].sum(skipna=True)
        crr_val   = sub["crr"].sum(skipna=True)
        return pd.Series([i10fg_val, tp_val, crr_val], index=[OUT_I10FG, OUT_TP, OUT_CRR])

    res = storm_bad.apply(agg_one, axis=1)
    
    # Write back to storms_data
    storms_data.loc[storm_bad.index, [OUT_I10FG, OUT_TP, OUT_CRR]] = res.values

return storms_data
