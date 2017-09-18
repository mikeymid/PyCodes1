import pandas as pd
import api_data_load
from IPython import embed

df = pd.read_csv('soil_data_48_locs.csv')
lat_lngs = list(zip(df.longitude.values.tolist(),
        df.latitude.values.tolist()))


ndf = api_data_load.get_plot_geometries_ll(lat_lngs)

ndf.to_csv('plot_info_48.csv')

