import geopandas as gpd
from pathlib import Path
import pandas as pd

def road_datasets_process(DATA_ROOT, df_ne_costal, counties):
    results = []
    
    for county_dir in DATA_ROOT.iterdir():
        if not county_dir.is_dir():
            continue
    
        county_name = county_dir.name
        county_fips = county_name.split("_")[0]   # 如 23001
    
        edges_path = county_dir / "edges" / "edges.shp"
        if not edges_path.exists():
            continue
    
        gdf = gpd.read_file(edges_path)
    
        if gdf.crs.to_epsg() == 4326:
            gdf = gdf.to_crs(epsg=3857)
    
        total_length_m = gdf.length.sum()
        total_length_km = total_length_m / 1000
        results.append({
            "county_fips": county_fips,
            "county_name": county_name,
            "road_length_km": total_length_km
        })
    
    df_roads = pd.DataFrame(results)
    counties["county_fips"] = counties["STATEFP"] + counties["COUNTYFP"]
    counties = counties.to_crs(epsg=3857)
    counties["area_km2"] = counties.area / 1e6
    df_final = counties.merge(df_roads, on="county_fips", how="left")
    df_final["road_density_km_per_km2"] = (
        df_final["road_length_km"] / df_final["area_km2"]
    )
    df_ne_costal["fips"] = df_ne_costal["fips"].astype(str).str.zfill(5)
    df_final["county_fips"] = df_final["county_fips"].astype(str).str.zfill(5)
    df_final_ne = df_final[df_final["county_fips"].isin(df_ne_costal["fips"])].copy()
    cols_keep = [
        "county_fips",
        "NAME",         
        "STATEFP",
        "road_length_km",
        "area_km2",
        "road_density_km_per_km2",
    ]
    road_density = df_final_ne[cols_keep]
    return road_density

