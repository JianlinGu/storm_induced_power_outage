import os
from pathlib import Path
from datetime import timedelta
import pandas as pd
import geopandas as gpd


from pathlib import Path
BASE_DIR = Path().resolve()

COUNTIES_SHP = (
    BASE_DIR
    / "data" / "raw" / "shapefiles"
    / "ne_counties" / "NE_coastal_counties.shp"
)

GRID_MAP_CSV = (
    BASE_DIR
    / "data" / "raw" / "shapefiles"
    / "era5_grid_to_fips.csv"
)

ERA5_DIR = (
    BASE_DIR
    / "data" / "raw" / "storm_intensity"
    / "era5_NE_coastal_county_hourly"
)


def build_grid_to_fips_mapping(
    counties_shp: str = COUNTIES_SHP,
    grid_map_csv: str = GRID_MAP_CSV,
    era5_dir: Path = ERA5_DIR,
    cache_to_disk: bool = True,
) -> pd.DataFrame:
    if cache_to_disk and os.path.exists(grid_map_csv):
        df_map = pd.read_csv(grid_map_csv)
        df_map["full_fips"] = df_map["full_fips"].astype(str).str.zfill(5)
        return df_map

    gdf_counties = gpd.read_file(counties_shp).to_crs(epsg=4326)

    if "GEOID" in gdf_counties.columns:
        gdf_counties["full_fips"] = gdf_counties["GEOID"].astype(str).str.zfill(5)
    else:
        gdf_counties["STATEFP"] = gdf_counties["STATEFP"].astype(str).str.zfill(2)
        gdf_counties["COUNTYFP"] = gdf_counties["COUNTYFP"].astype(str).str.zfill(3)
        gdf_counties["full_fips"] = (gdf_counties["STATEFP"] + gdf_counties["COUNTYFP"]).astype(str).str.zfill(5)

    sample_files = sorted(era5_dir.glob("data *.csv"))
    sample_file = sample_files[0]

    df_sample = pd.read_csv(sample_file, usecols=["latitude", "longitude"])
    df_grid = df_sample.drop_duplicates().reset_index(drop=True)

    gdf_grid = gpd.GeoDataFrame(
        df_grid,
        geometry=gpd.points_from_xy(df_grid["longitude"], df_grid["latitude"]),
        crs="EPSG:4326",
    )

    gdf_join = gpd.sjoin(
        gdf_grid,
        gdf_counties[["full_fips", "geometry"]],
        how="inner",
        predicate="within",
    )

    grid_to_fips = gdf_join[["latitude", "longitude", "full_fips"]].drop_duplicates()
    grid_to_fips["full_fips"] = grid_to_fips["full_fips"].astype(str).str.zfill(5)

    if cache_to_disk:
        os.makedirs(os.path.dirname(grid_map_csv), exist_ok=True)
        grid_to_fips.to_csv(grid_map_csv, index=False, encoding="utf-8-sig")

    return grid_to_fips

def era5_file_to_county_hourly_max_df(
    csv_path: Path,
    grid_map: pd.DataFrame,
) -> pd.DataFrame:
    df = pd.read_csv(
        csv_path,
        usecols=["valid_time", "latitude", "longitude", "tp", "i10fg", "crr"],
    )
    df = df.merge(
        grid_map[["latitude", "longitude", "full_fips"]],
        on=["latitude", "longitude"],
        how="inner",
    )
    df["valid_time"] = pd.to_datetime(df["valid_time"])
    df["full_fips"] = df["full_fips"].astype(str).str.zfill(5)

    # county × hour 只保留 max
    df_group = (
        df.groupby(["full_fips", "valid_time"], as_index=False)
          .agg(
              i10fg_max=("i10fg", "max"),
              tp_max=("tp", "max"),
              crr_max=("crr", "max"),
          )
    )
    return df_group

def build_storm_weather_features_max_total_48h_stream(
    df_storm: pd.DataFrame,
    grid_map: pd.DataFrame,
    era5_dir: Path = ERA5_DIR,
) -> pd.DataFrame:

    df_storm = df_storm.copy()
    need_cols = {"BEGIN_DATE_TIME", "STATE_FIPS", "CZ_FIPS", "EVENT_ID", "EPISODE_ID_LOC"}

    df_storm["BEGIN_DATE_TIME"] = pd.to_datetime(df_storm["BEGIN_DATE_TIME"], errors="coerce")

    df_storm["STATE_FIPS"] = df_storm["STATE_FIPS"].astype(float).astype(int).astype(str).str.zfill(2)
    df_storm["CZ_FIPS"]    = df_storm["CZ_FIPS"].astype(float).astype(int).astype(str).str.zfill(3)
    df_storm["full_fips"]  = (df_storm["STATE_FIPS"] + df_storm["CZ_FIPS"]).astype(str).str.zfill(5)

    df_storm["t0"] = df_storm["BEGIN_DATE_TIME"] - timedelta(hours=24)
    df_storm["t1"] = df_storm["BEGIN_DATE_TIME"] + timedelta(hours=24)

    df_storm = df_storm.reset_index(drop=True)
    df_storm["storm_idx"] = df_storm.index.astype(int)

    feat = pd.DataFrame({
        "storm_idx": df_storm["storm_idx"],
        "era_i10fg_max_total_48h": pd.NA,
        "era_tp_max_total_48h": pd.NA,
        "era_crr_max_total_48h": pd.NA,
    })

    current_i10fg = [None] * len(df_storm)
    current_tp    = [None] * len(df_storm)
    current_crr   = [None] * len(df_storm)

    all_files = sorted(era5_dir.glob("data *.csv"))

    for csv_path in all_files:

        df_ch = era5_file_to_county_hourly_max_df(csv_path, grid_map)


        t_min = df_ch["valid_time"].min()
        t_max = df_ch["valid_time"].max()

        mask = (df_storm["t1"] >= t_min) & (df_storm["t0"] <= t_max)
        sub = df_storm.loc[mask, ["storm_idx", "full_fips", "t0", "t1"]]

        if sub.empty:
            continue

        by_fips = {k: g for k, g in df_ch.groupby("full_fips", sort=False)}

        for r in sub.itertuples(index=False):
            w = by_fips.get(r.full_fips)
            if w is None:
                continue

            ww = w[(w["valid_time"] >= r.t0) & (w["valid_time"] <= r.t1)]
            if ww.empty:
                continue

            i10 = float(ww["i10fg_max"].max())
            tp  = float(ww["tp_max"].max())
            crr = float(ww["crr_max"].max())

            idx = int(r.storm_idx)

            if current_i10fg[idx] is None or i10 > current_i10fg[idx]:
                current_i10fg[idx] = i10
            if current_tp[idx] is None or tp > current_tp[idx]:
                current_tp[idx] = tp
            if current_crr[idx] is None or crr > current_crr[idx]:
                current_crr[idx] = crr

        del df_ch, by_fips

    feat["era_i10fg_max_total_48h"] = current_i10fg
    feat["era_tp_max_total_48h"]    = current_tp
    feat["era_crr_max_total_48h"]   = current_crr

    df_final = (
        df_storm
        .merge(feat, on="storm_idx", how="left")
        .drop(columns=["storm_idx", "t0", "t1"], errors="ignore")
    )
    return df_final


def run_all_stream(df_storm: pd.DataFrame) -> pd.DataFrame:
    grid_map = build_grid_to_fips_mapping(cache_to_disk=True)  # 想完全不落盘：改 False
    return build_storm_weather_features_max_total_48h_stream(df_storm=df_storm, grid_map=grid_map)


if __name__ == "__main__":
    raise RuntimeError('Import this instead')
