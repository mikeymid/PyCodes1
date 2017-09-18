import requests
import collections
import numpy as np
import pandas as pd
from tqdm import tqdm
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
from shapely.wkt import loads as load_wkt
import geopandas as gpd

from ssurgo_api.summarize import get_summary_df

def flatten(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


class Api(object):
    def __init__(self, client_id, username, password):
        '''
        Parameters
        ----------
        client_id - string
            The ID for the client that has been authorized for SSURGO API access.

        username - string
            username for the accessing user. For me this is the same username 
            that I use to log in to windows, but maybe other types of usernames
            can also work.

        password - string
            The password that corresponds to username
        '''
        self.client_id = client_id
        self.username = username
        self.password = password
        self.token = self.get_auth()

    def get_auth(self):
        '''
        Generate an access token
        '''

        auth_url="https://amp.monsanto.com/as/token.oauth2"
        headers = {'cache-control': "no-cache",
                   'content-type': "application/x-www-form-urlencoded"}
        credentials = {
                'url':auth_url,
                'client_id':self.client_id,
                'username':self.username,
                'password':self.password,
                'grant_type':'password'}

        response = requests.request("POST", auth_url, data=credentials, 
                headers=headers)
        token_data = response.json()

        return token_data


    def by_poly(self, poly_wkt, poly_buffer=0, debug=False):
        # {{{
        '''
        Get SSURGO results within the polygon specified by poly_coords

        Parameters
        ----------
        
        poly_coords - List
            should be passed in as a list of (long, lat) pairs in which the 
            first point is the same as the last point.

        poly_buffer - Float
            An ammount to expand the polygon. This is passed to Polygon.Buffer

        
        Returns
        -------
        (resp, df)
        resp 
            The response from the API.

        df - Pandas DataFrame
            a formatted version of the response
        '''
        token_data = self.get_auth()
        token = token_data['access_token']

        if poly_buffer != 0:
            poly = load_wkt(poly_wkt)
            poly = poly.buffer(poly_buffer)
            poly_wkt = poly.to_wkt()

        url = ('https://api01.agro.services/loc360/api/ssurgo/shape/wkt/' + 
                poly_wkt)

        headers = {'authorization': "Bearer {}".format(token),
                   'cache-control': "no-cache",
                   'content-type': "application/x-www-form-urlencoded"}

        resp = requests.request("GET", url, headers=headers)
        if debug:
            print(resp)

        # create the Geodataframe (with the polys) and the regular data frame
        #   (with the soil data)

        # Create the GeoDataFrame
        shapes = resp.json()['shapes']
        if debug:
            print(shapes)
        shapes_df = gpd.GeoDataFrame()
        for shape in shapes:
            data_id = shape['dataId']

            # get the polygon
            geometry_text = shape['geometryText']
            poly = load_wkt(geometry_text)
            shapes_df = shapes_df.append({'dataId':data_id, 'geometry':poly},
                    ignore_index=True)

        # Create the soil data DataFrame
        soil_df = pd.DataFrame()
        soil_data = resp.json()['data']
        for data_id in soil_data.keys():
            features = soil_data[data_id]
            for i, horizon in enumerate(
                    features['componentHorizons']):
                features['componentHorizon'+str(i+1)] = horizon

            features.pop('componentHorizons')

            result_dict = flatten(features)
            result_dict['dataId'] = data_id
            soil_df = soil_df.append(result_dict, ignore_index=True)

        # Merge the 2 data frames
        df = shapes_df.merge(soil_df, on='dataId')
            
        return resp, df # }}}


    def by_point(self, points):
        # {{{
        '''
        Parameters
        ----------
        points - list
            A list of (lng, lat) pairs
        '''
        # Get an authorization token
        token_data = self.get_auth()
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
        # }}}


    def poly_summary(self, poly_wkt, poly_buffer=0, depth_in=36):
        '''
        Get the average soil data for a poly
        '''
        resp, df = self.by_poly(poly_wkt, poly_buffer=poly_buffer)

        summary_df = get_summary_df(df, depth_in=depth_in, 
                cols_to_keep=['geometry'])
        
        weights = summary_df.area

        results_df = pd.DataFrame()
        for col in ['AWS', 'organic_matter', 'sand', 'clay', 'cec', 'ph']:
            vals = summary_df[col].values
            if set(vals) == set([None]):
                results_df[col] = None 
                continue

            try:
                indices = ~np.isnan(vals)
            except:
                print(vals)
            results_df[col] = [np.average(vals[indices], 
                    weights=weights[indices])]

        return results_df


