import pandas as pd
import geopandas as gpd
import shapely
import requests
import os
import location_selection

from IPython import embed


# Function for generating access_token
def get_auth():
    # {{{
    auth_url="https://amp.monsanto.com/as/token.oauth2"
    headers = {'cache-control': "no-cache",
               'content-type': "application/x-www-form-urlencoded"}
    credentials = {
            'url':auth_url,
            'client_id':os.environ['SRU_CLIENT_ID'],
            'username':os.environ['USERNAME'],
            'password':os.environ['PASSWORD'],
            'grant_type':'password'}

    response = requests.request("POST", auth_url, data=credentials, 
            headers=headers)
    token_data = response.json()

    return token_data #}}}

def get_plot_geometries(plot_ids):
    token_data = get_auth()
    token = token_data['access_token']

    url = "https://api01.agro.services/loc360/geoserver/v2/ows"

    headers = {'authorization': "Bearer {}".format(token),
               'cache-control': "no-cache",
               'content-type': "application/x-www-form-urlencoded"}

    querystring = {"service":"WFS",
                   "version":"2.0.0",
                   "request":"GetFeature",
                   "typeNames":"plots:plot_api",
                   "srsname":"EPSG:4326",
                   "outputFormat":"application/json",}
    
    # add the cql_filter to the querystring
    cql_filter = 'plot_id in ('
    for plot_id in plot_ids:
        cql_filter += str(plot_id) + ', '

    cql_filter = cql_filter[:-2] + ')'
    querystring['cql_filter'] = cql_filter

    payload = "="

    resp = requests.request("GET", url, data=payload, headers=headers, 
            params=querystring)
    
    crs = {'init': 'epsg:4326'}
    geometries_df = gpd.GeoDataFrame(crs=crs)

    df_plot_ids = []
    geometries = []
    sources = []
    for item in resp.json()['features']:
        plot_id = item['properties']['plot_id']
        coords = item['geometry']['coordinates'][0]
        poly = shapely.geometry.Polygon(coords)
        df_plot_ids.append(plot_id)
        geometries.append(poly)
        sources.append(item['properties']['data_source'])

    geometries_df['plot_id'] = df_plot_ids
    geometries_df['geometry'] = geometries
    geometries_df['centroid'] = geometries_df.geometry.centroid
    geometries_df['longitude'] = [p.x for p in geometries_df.centroid.values]
    geometries_df['latitude'] = [p.y for p in geometries_df.centroid.values]
    geometries_df['source'] = sources

    return geometries_df
    
#################################################
yield_fpath3 = 'yield_hybrids_2016_bu.csv'
df3 = pd.read_csv(yield_fpath3)    
df3 = df3[~df3.BREEDING_PROGRAM.str.contains('TECH DEV')]
df3 = df3[~df3.LOCATION.str.endswith(', CAN')]
df3 = df3['BR_LOC_REF_ID']
print(df3.nunique(dropna=True))

df4 = list(set(df3))

df4 = df4[:4]
print(df4)
##################################################

yield_fpath = 'all_yield.csv'
df2 = pd.read_csv(yield_fpath)
df2.to_csv('test')
plot_ids_df2 = df2.PLOT_ID.values.tolist()

if __name__ == '__main__':
    plot_ids = plot_ids_df2
    myList = df4

    for fld_id in myList:
        idf = df2[df2.BR_LOC_REF_ID == fld_id]
        plot_ids = idf.PLOT_ID.unique().tolist()
        plot_ids = [int(pid) for pid in plot_ids]
        print('Num plot_ids for ' + fld_id + ': ' + str(len(plot_ids)))


        first_plot = 0
        last = 200

        num_plot_geometries_found = 0

        while first_plot < len(plot_ids):
            sub_plot_ids = plot_ids[first_plot:last]
            df = get_plot_geometries(sub_plot_ids)
            num_plot_geometries_found += len(df)

            df.to_csv(fld_id + str(first_plot) + '.csv')
            first_plot += 200
            last += 200

    print('Num geometries returned by API: ' + 
            str(num_plot_geometries_found))
        

##################









