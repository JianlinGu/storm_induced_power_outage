import pandas as pd
import geopandas as gpd

def strom_impact_location_exposure(gdf_urban, storms_data):
    gdf_storm = gpd.GeoDataFrame(
        storms_data,
        geometry=gpd.points_from_xy(storms_data["LONGITUDE"], storms_data["LATITUDE"]),
        crs="EPSG:4326"
    )
    
    gdf_join = gpd.sjoin(
        gdf_storm,
        gdf_urban[["geometry"]],
        how="left",
        predicate="within"
    )
    
    gdf_join["urban_flag"] = gdf_join["index_right"].notna().astype(int)
    
    group_cols = ["CZ_FIPS", "EPISODE_ID_LOC"]
    
    agg = (
        gdf_join.groupby(group_cols)
        .agg(
            n_points=("urban_flag", "size"),
            n_urban=("urban_flag", "sum"),
            urban_ratio=("urban_flag", "mean")  # 城市命中比例
        )
        .reset_index()
    )
    
    storms_data = storms_data.merge(agg, on=group_cols, how="left")
    storms_data[["n_points", "n_urban", "urban_ratio"]] = storms_data[["n_points", "n_urban", "urban_ratio"]].fillna(0)
    return storms_data