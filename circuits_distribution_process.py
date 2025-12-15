import pandas as pd
import numpy as np

def circuits_distribution_process(dist_sys, serv_terr, road_density, master_data):
    dist_sys = dist_sys.copy()
    serv_terr = serv_terr.copy()
    road_density = road_density.copy()

    dist_sys['Distribution Circuits'] = (
        dist_sys['Distribution Circuits']
          .astype(str)
          .str.replace(',', '', regex=False)
          .replace('.', np.nan)
          .astype(float)
    )
    

    dist_sys = (
        dist_sys.sort_values(['Utility Number', 'Data Year'])
                .drop_duplicates(subset=['Utility Number', 'State'], keep='last')
    )
    

    dist_sys_small = dist_sys[['Utility Number', 'State', 'Distribution Circuits']]
    

    serv_small = (
        serv_terr[['Utility Number', 'State', 'County']]
          .drop_duplicates()
    )

    state_fips_to_abbr = {
        1: "AL", 2: "AK", 4: "AZ", 5: "AR", 6: "CA",
        8: "CO", 9: "CT", 10: "DE", 11: "DC", 12: "FL",
        13: "GA", 15: "HI", 16: "ID", 17: "IL", 18: "IN",
        19: "IA", 20: "KS", 21: "KY", 22: "LA", 23: "ME",
        24: "MD", 25: "MA", 26: "MI", 27: "MN", 28: "MS",
        29: "MO", 30: "MT", 31: "NE", 32: "NV", 33: "NH",
        34: "NJ", 35: "NM", 36: "NY", 37: "NC", 38: "ND",
        39: "OH", 40: "OK", 41: "OR", 42: "PA", 44: "RI",
        45: "SC", 46: "SD", 47: "TN", 48: "TX", 49: "UT",
        50: "VT", 51: "VA", 53: "WA", 54: "WV", 55: "WI",
        56: "WY"
    }
    
    road_density['STATEFP'] = road_density['STATEFP'].astype(int)
    road_density['State'] = road_density['STATEFP'].map(state_fips_to_abbr)
    
    road_small = road_density[[
        'county_fips', 'NAME', 'State', 'road_density_km_per_km2'
    ]].rename(columns={'NAME': 'County'})

    road_small['County'] = road_small['County'].str.strip()
    serv_small['County'] = serv_small['County'].str.strip()

    util_cnty = serv_small.merge(
        road_small,
        on=['State', 'County'],
        how='left',
        validate='m:1'
    )

    mask_na = util_cnty['road_density_km_per_km2'].isna()
    if mask_na.any():
        util_cnty['road_density_km_per_km2'] = util_cnty.groupby('State')['road_density_km_per_km2']\
            .transform(lambda x: x.fillna(x.mean()))
        util_cnty['road_density_km_per_km2'] = util_cnty['road_density_km_per_km2'].fillna(
            util_cnty['road_density_km_per_km2'].mean()
        )

    util_cnty = util_cnty.merge(
        dist_sys_small,
        on=['Utility Number', 'State'],
        how='left',
        validate='m:1'
    )

    util_cnty = util_cnty.dropna(subset=['Distribution Circuits'])
    grp = util_cnty.groupby('Utility Number')['road_density_km_per_km2']
    sum_density = grp.transform('sum')
    zero_sum = (sum_density == 0) | sum_density.isna()
    util_cnty.loc[~zero_sum, 'weight'] = \
        util_cnty.loc[~zero_sum, 'road_density_km_per_km2'] / sum_density[~zero_sum]
    util_cnty.loc[zero_sum, 'weight'] = \
        1.0 / util_cnty.loc[zero_sum].groupby('Utility Number')['Utility Number'].transform('count')
    util_cnty['circuits_county_weighted'] = \
        util_cnty['Distribution Circuits'] * util_cnty['weight']

    county_circuits = (
        util_cnty
          .groupby(['county_fips', 'State', 'County'], as_index=False)['circuits_county_weighted']
          .sum()
          .rename(columns={'circuits_county_weighted': 'circuits_total'})
    )

    print(county_circuits.head())
    print(county_circuits.shape)

    county_circuits_clean = county_circuits.copy()
    county_circuits_clean['County_up'] = (
        county_circuits_clean['County']
        .str.upper()
        .str.strip()
    )

    master_data['CZ_NAME'] = master_data['CZ_NAME'].str.upper().str.strip()

    master_data = master_data.merge(
        county_circuits_clean[['County_up', 'circuits_total']],
        left_on='CZ_NAME',
        right_on='County_up',
        how='left'
    )

    master_data = master_data.drop(columns=['County_up'])
    master_data = master_data.rename(
        columns={'circuits_total': 'weighted_number_of_circuits'}
    )
    return master_data
