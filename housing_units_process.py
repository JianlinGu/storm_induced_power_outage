import pandas as pd
import numpy as np
    
def housing_units_process(df_hu20202024, df_hu20102020, ne_coastal):
    ne_coastal['fips_str'] = ne_coastal['fips'].astype(str).str.zfill(5)
    hu10 = df_hu20102020[df_hu20102020['SUMLEV'] == 50].copy()
    hu10['fips_str'] = (
        hu10['STATE'].astype(str).str.zfill(2) +
        hu10['COUNTY'].astype(str).str.zfill(3)
    )
    hu10 = hu10[hu10['fips_str'].isin(ne_coastal['fips_str'])].copy()
    rename_10_19 = {f'HUESTIMATE{y}': str(y) for y in range(2010, 2020)}
    cols_10_19 = ['fips_str'] + list(rename_10_19.keys())
    
    hu10_years = (hu10[cols_10_19]
                  .rename(columns=rename_10_19))
    if 'loaction' in df_hu20202024.columns:
        df_hu20202024 = df_hu20202024.rename(columns={'loaction': 'location'})
    hu20 = df_hu20202024.copy()

    hu20['location'] = hu20['location'].str.lstrip('.').str.strip()
    tmp = hu20['location'].str.split(',', n=1, expand=True)
    hu20['county'] = tmp[0].str.replace(' County', '', regex=False).str.strip()
    hu20['state'] = tmp[1].str.strip()
    hu20 = hu20[hu20['state'].notna()].copy()

    for y in range(2020, 2025):
        col = str(y)
        hu20[col] = (hu20[col]
                     .astype(str)
                     .str.replace(',', '', regex=False)
                     .astype('Int64'))

    hu20_ne = ne_coastal.merge(
        hu20[['county', 'state', '2020', '2021', '2022', '2023', '2024']],
        left_on=['county', 'state'],
        right_on=['county', 'state'],
        how='left'
    )
    
    hu20_ne['fips_str'] = hu20_ne['fips'].astype(str).str.zfill(5)
    hu20_years = hu20_ne[['fips_str', '2020', '2021', '2022', '2023', '2024']].copy()

    hu_2010_2024 = (ne_coastal[['state', 'county', 'fips_str']]
                    .merge(hu10_years, on='fips_str', how='left')
                    .merge(hu20_years, on='fips_str', how='left'))
    cols_order = (
        ['state', 'county', 'fips_str'] +
        [str(y) for y in range(2010, 2025)]
    )
    hu_2010_2024 = hu_2010_2024[cols_order]
    return hu_2010_2024