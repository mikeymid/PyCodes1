import pandas as pd
import geopandas as gpd
import shapely
import requests
import os
from tqdm import tqdm

from IPython import embed

import collections

def flatten(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

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

def get_plot_geometries(plot_ids, batch_size=200):
    # {{{
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
                   "outputFormat":"application/json"}

    batch_start = 0
    batch_stop = batch_start + batch_size

    crs = {'init': 'epsg:4326'}
    results_df = gpd.GeoDataFrame(crs=crs)
    
    while batch_start < len(plot_ids):
        batch_plot_ids = plot_ids[batch_start:batch_stop]
    
        # add the cql_filter to the querystring
        cql_filter = 'plot_id in ('
        for plot_id in batch_plot_ids:
            cql_filter += str(plot_id) + ', '

        cql_filter = cql_filter[:-2] + ')'
        querystring['cql_filter'] = cql_filter

        payload = "="

        resp = requests.request("GET", url, data=payload, headers=headers, 
                params=querystring)
        
        crs = {'init': 'epsg:4326'}
        batch_df = gpd.GeoDataFrame(crs=crs)

        df_plot_ids = []
        geometries = []
        sources = []
        field_ids = []
        for item in resp.json()['features']:
            plot_id = item['properties']['plot_id']
            coords = item['geometry']['coordinates'][0]
            poly = shapely.geometry.Polygon(coords)
            df_plot_ids.append(plot_id)
            geometries.append(poly)
            sources.append(item['properties']['data_source'])
            field_ids.append(item['properties']['field_id'])

        batch_df['plot_id'] = df_plot_ids
        batch_df['geometry'] = geometries
        batch_df['centroid'] = batch_df.geometry.centroid
        batch_df['latitude'] = [p.y for p in batch_df.centroid.values]
        batch_df['longitude'] = [p.x for p in batch_df.centroid.values]
        batch_df['source'] = sources
        batch_df['field_id'] = field_ids

        batch_start += batch_size
        batch_stop += batch_size


        results_df = results_df.append(batch_df, ignore_index=True)
    n_found = results_df.plot_id.nunique()
    pct_found = round(n_found/float(len(plot_ids)),1)
    print('Found data for ' + str(n_found) + ' of ' + 
            str(len(plot_ids)) + ' plot ids (' + str(pct_found) + '%)')

    return results_df # }}}

def get_plot_geometries_ll(lat_lngs, batch_size=50):
    # {{{
    token_data = get_auth()
    token = token_data['access_token']

    url = "https://api01.agro.services/loc360/geoserver/v2/ows"

    headers = {'authorization': "Bearer {}".format(token),
               'cache-control': "no-cache",
               'content-type': "application/x-www-form-urlencoded"}

    querystring = {"service":"WFS",
                   "version":"1.0.0",
                   "request":"GetFeature",
                   "typeNames":"plots:plot_api",
                   "srsname":"EPSG:4326",
                   "outputFormat":"application/json"}

    batch_start = 0
    batch_stop = batch_start + batch_size

    crs = {'init': 'epsg:4326'}
    results_df = gpd.GeoDataFrame(crs=crs)

    #cql_filter = 'INTERSECTS(geom, POINT (-84.214661 31.747830))'
    
    while batch_start < len(lat_lngs):
        batch_lat_lngs = lat_lngs[batch_start:batch_stop]

        cql_filter = 'INTERSECTS(geom, MultiPoint ('
        for lat_lng in lat_lngs:
            lng = lat_lng[0]
            lat = lat_lng[1]
            cql_filter += '(' + str(lng) + ' ' + str(lat) + '), '

        
        # Remove the extra comma and space and close the parentheses
        cql_filter = cql_filter[:-2] + '))'
        print(cql_filter)
    
        querystring['cql_filter'] = cql_filter

        payload = "="

        resp = requests.request("GET", url, data=payload, headers=headers, 
                params=querystring)
        
        crs = {'init': 'epsg:4326'}
        batch_df = gpd.GeoDataFrame(crs=crs)

        df_plot_ids = []
        geometries = []
        sources = []
        field_ids = []
        seasons = []
        for item in resp.json()['features']:
            plot_id = item['properties']['plot_id']
            coords = item['geometry']['coordinates'][0]
            poly = shapely.geometry.Polygon(coords)
            df_plot_ids.append(plot_id)
            geometries.append(poly)
            sources.append(item['properties']['data_source'])
            field_ids.append(item['properties']['field_id'])
            seasons.append(item['properties']['season_year'])

        batch_df['plot_id'] = df_plot_ids
        batch_df['geometry'] = geometries
        batch_df['plot_centroid'] = batch_df.geometry.centroid
        batch_df['plot_latitude'] = [p.y for p in batch_df.centroid.values]
        batch_df['plot_longitude'] = [p.x for p in batch_df.centroid.values]
        batch_df['source'] = sources
        batch_df['field_id'] = field_ids
        batch_df['season'] = seasons
        batch_df['queried_latitude'] = lat
        batch_df['queried_longitude'] = lng

        batch_start += batch_size
        batch_stop += batch_size

        results_df = results_df.append(batch_df, ignore_index=True)

    n_found = len(results_df.groupby(['queried_latitude', 'queried_longitude']
            ).count()) 
    pct_found = round(100*n_found/float(len(lat_lngs)),1)
    print('Found data for ' + str(n_found) + ' of ' + 
            str(len(lat_lngs)) + ' coords (' + str(pct_found) + '%)')

    return results_df # }}}

def ssurgo_by_poly(poly):
    # {{{
    '''
    poly should be passed in as a list of (long, lat) pairs, in which the first
    point is the same as the last point.
    '''
    token_data = get_auth()
    token = token_data['access_token']

    url = "https://api01.agro.services/loc360/api/ssurgo/shape/wkt/POLYGON(("
    for lng, lat in poly:
        url += str(lng) + ' ' + str(lat) + ', '

    # Remove the extra comma and space and close the parentheses
    url = url[:-2] + '))'

    headers = {'authorization': "Bearer {}".format(token),
               'cache-control': "no-cache",
               'content-type': "application/x-www-form-urlencoded"}

    resp = requests.request("GET", url, headers=headers)

    embed() # }}}

def ssurgo_by_point(points):
    # {{{
    '''
    poly should be passed in as a list of (long, lat) pairs, in which the first
    point is the same as the last point.
    '''
    token_data = get_auth()
    token = token_data['access_token']

    base_url = ('https://api01.agro.services:443/loc360/api/ssurgo/' +
            'geojson/feature/point/')

    headers = {'authorization': "Bearer {}".format(token),
           'cache-control': "no-cache",
           'content-type': "application/x-www-form-urlencoded"}

    bad_coords = pd.DataFrame()

    df = pd.DataFrame()

    result_dicts = []

    for point in tqdm(points, desc='API calls'):
        lng = point[0]
        lat = point[1]
        url = base_url + 'latitude/' + str(lat) + '/longitude/' + str(lng)

        resp = requests.request("GET", url, headers=headers)

        # Check if we got useful results
        if not 'features' in resp.json().keys():
            bad_coords = bad_coords.append({'lat':lat, 'lng':lng}, 
                    ignore_index=True)
            continue

        features = resp.json()['features'][0]

        for i, horizon in enumerate(
                features['properties']['componentHorizons']):
            features['properties']['componentHorizon'+str(i+1)] = horizon

        features['properties'].pop('componentHorizons')

        result_dict = flatten(features)

        result_dicts.append(result_dict)

    df = df.append(result_dicts, ignore_index=True)

    bad_coords.to_csv('ssurgo_bad_coords.csv', index=False)
    df.to_csv('ssurgo_data.csv', index=False)
    return df

def soil_samples():
    token_data = get_auth()
    token = token_data['access_token']

    url = "https://api.monsanto.com:443/soilsamples/v2/published-imports?datefilter=2013-09-01"

    headers = {'authorization': "Bearer {}".format(token),
               'cache-control': "no-cache",
               'content-type': "application/x-www-form-urlencoded"}

    querystring = {"service":"WFS",
                   "version":"1.0.0",
                   "request":"GetFeature",
                   "typeNames":"plots:plot_api",
                   "srsname":"EPSG:4326",
                   "outputFormat":"application/json"}
    

if __name__ == '__main__':
    test_case = 'plot_ids'
    test_case = 'plot_points'
    test_case = 'ssurgo'
    test_case = 'ssurgo_points'

    if test_case == 'plot_ids':
        plot_ids = [879542288855964, 879542288855948, 879542288856327]
        df = get_plot_geometries(plot_ids)

    elif test_case == 'plot_points':
        lat_longs = [
                #(-89.19827779, 39.62827287),
                (-93.6408798, 41.8460672)
                #(-84.214725, 31.747832),
                #(-84.214955, 31.743404)
                ]
        df = get_plot_geometries_ll(lat_longs)

    elif test_case == 'ssurgo':
        mnpx_poly = [
                (-93.065293, 43.899201),
                (-93.057131, 43.899207),
                (-93.056915, 43.895801),
                (-93.065239, 43.895813),
                (-93.065293, 43.899201)
                ]
        ssurgo_by_poly(mnpx_poly)

    elif test_case == 'ssurgo_points':
        coords_df = pd.read_csv('test_grid_csv.csv')
        #coords_df = coords_df[:50]
        lats = coords_df.centLat.values.tolist()
        lngs = coords_df.centLong.values.tolist()
        coords = zip(lngs,lats)
        test_coords = [
                (-92, 38.5),
                ]
        test_coords = [
                (-90, 39),
                ]
        test_coords = [
                (-92.82341792, 43.84857822),
                (-92.8233806, 43.84857817)
                ]
        df = ssurgo_by_point(coords)

    embed()
        




