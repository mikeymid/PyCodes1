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
    pct_found = round(100 * n_found/float(len(plot_ids)),1)
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
    
#################################################
yield_fpath3 = 'yield_hybrids_2016_bu.csv'
df3 = pd.read_csv(yield_fpath3)    
#df3 = df3[df3.BREEDING_PROGRAM.str.contains('WILLIAM JUSTIN GARRETT')]
#df3 = df3[~df3.BREEDING_PROGRAM.str.contains('TECH DEV')]
#df3 = df3[df3.LOCATION.str.endswith(', CAN')]

df3 = df3['BR_LOC_REF_ID']
print(df3.nunique(dropna=True))

df4 = list(set(df3))

#df4 = df4[:4]
print(df4)
##################################################

yield_fpath = 'yield_hybrids_2016_bu.csv'
df2 = pd.read_csv(yield_fpath)
#df2.to_csv('test')
plot_ids_df2 = df2.PLOT_ID.values.tolist()

if __name__ == '__main__':
    plot_ids = plot_ids_df2
    myList = df4
    
    df = pd.DataFrame()

    for fld_id in myList:
        idf = df2[df2.BR_LOC_REF_ID == fld_id]
        plot_ids = idf.PLOT_ID.unique().tolist()
        plot_ids = [int(pid) for pid in plot_ids]
        print('Num plot_ids for ' + fld_id + ': ' + str(len(plot_ids)))
        temp_df = get_plot_geometries(plot_ids)
        temp_df['Location'] = fld_id
        
        # Add yield data
        yld_df = idf[['PLOT_ID', 'NUM_VALUE']]
        temp_df = pd.merge(temp_df, yld_df, left_on='plot_id', 
                right_on='PLOT_ID', how='inner')
        temp_df.rename(columns={'NUM_VALUE':'Yield'}, inplace=True)
        df = df.append(temp_df, ignore_index=True)
        
    df = df[df.source == 'FIELD_DRIVE']
    df.to_csv('output1.csv')
         

    #print('Num geometries returned by API: ' + 
            #str(num_plot_geometries_found))
        

##################









