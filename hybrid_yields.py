import pandas as pd
import geopandas as gpd
import numpy as np
import os

from IPython import embed

# merge yld file with shapefile

def load_data():
    shapefile_fpaths = []

    fnames = [
        '879549362670933.shp',
        '879549362672620.shp',
        '879549363549769.shp',
        '879549016304644.shp']

    gdf = gpd.GeoDataFrame()
    for fname in fnames:
        fpath = os.path.join(os.getcwd(), 'kyst','shp', fname)
        temp_gdf = gpd.read_file(fpath)
        #embed()
        #a=b
    


if __name__ == '__main__':
    load_data()
