
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from geopy.distance import great_circle

#from pyKriging.krige import kriging
from pykrige.ok import OrdinaryKriging
from multiprocessing import Pool

import location_selection
from IPython import embed
from matplotlib.colors import LinearSegmentedColormap



def make_fig(soil_df, feat='ph', flds=['MNCF','MNWX','MNPX'], 
    # {{{
        variogram='linear',
        min_n_samples_for_kriging=20, show_fig=False, max_samples=500,
        grid_resolution=10, nlags=6):
    # --------- Visual settings ----------------
    l_font_size = 20
    s_font_size = 8


    # --------- Layout Settings ----------------
    top_padding = 0.05
    bottom_padding = 0.02
    left_padding = 0.02
    right_padding = 0.05
    vspace = 0.008
    hspace = 0.008
    #minor_vspace = 0.002
    #minor_hspace = 0.002

    dark_red = (0.5,0,0)
    red = (1,0,0)
    dark_green = (0,0.5,0)
    green = (0,1,0)
    light_blue = (0.5,0.5,1)
    dark_blue = (0,0,0.75)
    orange = (1,0.65,0)
    yellow = (1,1,0)

    feat_colors = {
            'ph':{'break_points':[5, 6, 7.5, 8, 8.5],
                'colors':[
                    (0.5,0,0), # Dark red
                    (1,0,0), # Red
                    (0,0.5,0), # Dark Green
                    (0,1,0), # Bright Green
                    (0.5,0.5,1), # Light Blue
                    (0,0,0.75), # Dark Blue 
                    ]},

            'organicMatter':{'break_points':[1, 2, 3, 4],
                'colors':[
                    (1,0,0), # Red
                    (1,0.65,0), # orange
                    (1,1,0), # Yellow
                    (0,0.5,0), # dark green
                    (0,0,0.75), # Dark Blue
                    ]},

            'p':{'break_points':[10, 15, 26, 35, 45],
                'colors':[
                    red,
                    yellow,
                    orange,
                    green,
                    light_blue,
                    dark_blue,
                    ]},

            'k':{'break_points':[60,100, 140, 200],
                'colors':[
                    red,
                    yellow,
                    orange,
                    green,
                    light_blue,
                    ]},

            's':{'break_points':[7.5,15],
                'colors':[
                    red,
                    green,
                    light_blue,
                    ]},

            'zn':{'break_points':[0.5,1,1.5,2,3],
                'colors':[
                    red,
                    yellow,
                    orange,
                    green,
                    light_blue,
                    dark_blue
                    ]},

            'cec':{'break_points':[10, 20],
                'colors':[
                    red,
                    green,
                    dark_blue
                    ]},

            'Yield':{'break_points':[150, 175, 200, 230, 250],
                'colors':[
                    (0.5,0,0), # Dark red
                    (1,0.5,0.16), # Orange
                    (1,1,0), # Yellow
                    (0,1,0), # Bright Green
                    (0,0,0.75), # Dark Blue
                    (0.898,0,0.663), # Pink
                    ]},

            }

    colors = feat_colors[feat]['colors']
    color_break_points = feat_colors[feat]['break_points']

    # add colors to the df
    soil_df['color'] = soil_df[feat].apply(lambda x:get_color_for_val(x,
            colors, color_break_points))

    soil_df.dropna(subset=[feat], inplace=True)

    
    #flds = list(soil_df.fieldId.value_counts().index)
    flds = list(soil_df.fieldId.unique())
    #flds.sort(reverse=True)
    flds.sort()
    #flds = flds[:23]
    #flds = flds[:40]
    print(str(len(flds)) + ' fields')

    fig_size_in = (24,24)

    num_rows = 6
    num_cols = 8

    num_subplots_per_fig = num_rows * num_cols

    ax_height = (1 - (num_rows-1)*vspace -
            top_padding - bottom_padding)/ float(num_rows)
    krig_height = ax_height
    ax_width = (1 - left_padding - right_padding - (num_cols-1)*hspace
            ) / float(num_cols)

    min_feat = soil_df[feat].min()
    max_feat = soil_df[feat].max()

    min_x = 0
    max_x = 0
    min_y = 0
    max_y = 0
    
    figs = {}
    for i, fld in tqdm(enumerate(flds), desc='Fields'):
        fig_num = i // num_subplots_per_fig + 1
        subplot_num = i % num_subplots_per_fig
        print('Field ' + str(i) + ') ' + fld)
        if subplot_num == 0: #this is the first fld for the figure
            fig = plt.figure()
            figs[fig_num] = fig

        row_num = subplot_num // num_cols
        col_num = subplot_num % num_cols

        is_left_ax = col_num == 0
        is_bottom_ax = row_num == num_rows - 1
        
        # create the axes
        left = left_padding + col_num*(ax_width+hspace)
        top = 1 - top_padding - row_num*(ax_height+vspace)
        bottom = top - ax_height

        rect = (left, bottom, ax_width, ax_height)
        if i == 0:
            ax = fig.add_axes(rect)
            first_ax = ax
            shared_ax = first_ax
        else:
            shared_ax = first_ax
            shared_ax = None
            ax = fig.add_axes(rect, sharex=shared_ax, sharey=shared_ax)

        fld_df = soil_df[soil_df.fieldId == fld]
        
        if len(fld_df) > max_samples:
            fld_df = fld_df.sample(max_samples)

        avg_lat = fld_df.latitude.mean()
        avg_lon = fld_df.longitude.mean()

        fld_df['x'] =fld_df.apply(lambda row: great_circle(
            (row['latitude'], row['longitude']), (row['latitude'], avg_lon)).m
            * np.sign(row.longitude - avg_lon), axis=1)

        fld_df['y'] =fld_df.apply(lambda row: great_circle(
            (row['latitude'], row['longitude']), (avg_lat, row.longitude)).m
            * np.sign(row.latitude - avg_lat), axis=1)

        num_samples = len(fld_df)
        do_kriging = num_samples >= min_n_samples_for_kriging
        do_kriging = False
        #lat = fld_df['latitude'].values
        #lon = fld_df['longitude'].values
        x = fld_df.x.values
        y = fld_df.y.values
        feat_vals = fld_df[feat].values

        # update the min and max x and y
        min_x = min(min_x, min(x))
        max_x = max(max_x, max(x))
        min_y = min(min_y, min(y))
        max_y = max(max_y, max(y))

        if do_kriging:
            do_and_plot_kriging(x, y, feat_vals, ax, colors, 
                    color_break_points, variogram=variogram, 
                    grid_resolution=grid_resolution, nlags=nlags)
        else:
            c = np.vstack(fld_df.color.values)
            if True:#num_samples < 500000:
                s = (100/float(np.sqrt(num_samples)) * 10 * 
                        np.mean([ax_width, ax_height]))
            else:
                s = 10 * np.mean([ax_width, ax_height])
            ax.scatter(x, y, c=c, edgecolor='none', s=s, alpha=1.0)

        ax.set_xlim(fld_df.x.min(), fld_df.x.max())
        ax.set_ylim(fld_df.y.min(), fld_df.y.max())
        ax.set_aspect('equal')
        ax.text(0.02, 0.98, (fld), verticalalignment='top', alpha=0.25,
                fontsize=s_font_size, transform=ax.transAxes)
        #font size for the fields - manipulate alpha / opacity

        #if not is_bottom_ax:
        plt.setp(ax.get_xticklabels(), visible=False)

        #if not is_left_ax:
        plt.setp(ax.get_yticklabels(), visible=False)


    if shared_ax == first_ax:
        first_ax.set_xlim(min_x, max_x)
        min_y_lim = min(first_ax.get_ylim()[0], min_y)
        max_y_lim = max(first_ax.get_ylim()[1], max_y)
        first_ax.set_ylim(min_y_lim, max_y_lim)

    # Add a colorbar
    cbar_left = 0.96
    cbar_bottom = 0.5
    cbar_width = 0.01
    cbar_height = 1 - cbar_bottom - top_padding
    cbar_rect = (cbar_left, cbar_bottom, cbar_width, cbar_height)  
    cbar_ax = fig.add_axes(cbar_rect)

    scale_range = np.ptp(np.array(color_break_points))
    cbar_min = color_break_points[0]-scale_range*0.15
    cbar_max = color_break_points[-1]+scale_range*0.15
    cbar_ax.axhspan(ymin=cbar_min, ymax=color_break_points[0], color=colors[0])

    for i in range(len(color_break_points)-1):
        ymin = color_break_points[i]
        ymax = color_break_points[i+1]
        cbar_ax.axhspan(ymin=ymin, ymax=ymax, color=colors[i+1])

    cbar_ax.axhspan(ymin=color_break_points[-1], ymax=cbar_max, 
            color=colors[-1])

    cbar_ax.set_xticks([])
    cbar_ax.set_yticks(color_break_points)
    cbar_ax.yaxis.tick_right()
    cbar_ax.yaxis.set_label_position('right')
    cbar_ax.set_ylabel(feat)
    cbar_ax.set_ylim(cbar_min, cbar_max)

    for fig_num in figs.keys():
        fig = figs[fig_num]
        fig.set_size_inches(fig_size_in)
        fig.savefig(feat + '_' + str(fig_num) + '.png', dpi=300)

    print(min_feat)
    print(max_feat)

    with open('dominostats.json', 'w') as f:
        stats_dict = {
                'feat':feat,
                'veriogram': variogram,
                'grid_resolution': grid_resolution,
                'nlags': nlags,
                'max_samples': max_samples}
        f.write(json.dumps(stats_dict))
    
    if show_fig:
        plt.show() # }}}
    
def do_and_plot_kriging_pykriging(X, y, ax, colors, break_points, nlags=6):
    # {{{
    xmin, ymin = X.min(axis=0)
    xmax, ymax = X.max(axis=0)
    k = kriging(X, y)
    k.train()

    numiter = 0
    for i in range(numiter):  
        print('Infill iteration {0} of {1}....'.format(i + 1, numiter))
        newpoints = k.infill(1)
        for point in newpoints:
            k.addPoint(point, k.predict(point))
        k.train()

    samplePoints = list(zip(*k.X))

    # Create a set of data to plot
    plotgrid = 10
    x = np.linspace(k.normRange[0][0], k.normRange[0][1], num=plotgrid)
    y = np.linspace(k.normRange[1][0], k.normRange[1][1], num=plotgrid)

    # x = np.linspace(0, 1, num=plotgrid)
    # y = np.linspace(0, 1, num=plotgrid)
    X, Y = np.meshgrid(x, y)

    # Predict based on the optimized results

    zs = np.array([k.predict([x,y]) for x,y in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(X.shape)

    # create colored Z
    colored_Z = np.zeros((Z.shape[0], Z.shape[1], 3))
    for row_num, row in enumerate(Z):
        for col_num, val in enumerate(row):
            for color, break_point in zip(colors, break_points):
                if val < break_point:
                    colored_Z[row_num, col_num] = color
                    break

    ax.imshow(colored_Z, interpolation='none', extent=[xmin, xmax, ymin, ymax])
    # Z = (Z*(k.ynormRange[1]-k.ynormRange[0]))+k.ynormRange[0]

    #Calculate errors
    #zse = np.array([k.predict_var([x,y]) for x,y in zip(np.ravel(X), 
    #    np.ravel(Y))])
    #Ze = zse.reshape(X.shape)

    '''
    spx = (k.X[:,0] * (k.normRange[0][1] - k.normRange[0][0])
            ) + k.normRange[0][0]
    spy = (k.X[:,1] * (k.normRange[1][1] - k.normRange[1][0])
            ) + k.normRange[1][0]
    contour_levels = 25

    CS = ax.contourf(X,Y,Z,contour_levels,zorder=1)
    ax.plot(spx, spy,'ow', zorder=3, alpha=0.1)
    '''
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    #plt.colorbar(ax=ax)

    return None # }}}


def do_and_plot_kriging(x, y, feat_vals, ax, colors, break_points,
    # {{{
        variogram='linear', grid_resolution=10, nlags=6):
    xmin = x.min()
    xmax = x.max()
    ymin = y.min()
    ymax = y.max()


    k = OrdinaryKriging(x, y, feat_vals, variogram_model=variogram,
            verbose=True, enable_plotting=False, nlags=nlags)

    grid_x = np.arange(xmin, xmax, grid_resolution)
    grid_y = np.arange(ymin, ymax, grid_resolution)

    z, ss = k.execute('grid', grid_x, grid_y)

    # Create a set of data to plot
    #plotgrid = 10
    #x = np.linspace(k.normRange[0][0], k.normRange[0][1], num=plotgrid)
    #y = np.linspace(k.normRange[1][0], k.normRange[1][1], num=plotgrid)

    # x = np.linspace(0, 1, num=plotgrid)
    # y = np.linspace(0, 1, num=plotgrid)
    #X, Y = np.meshgrid(x, y)

    # Predict based on the optimized results

    #zs = np.array([k.predict([x,y]) for x,y in zip(np.ravel(X), np.ravel(Y))])
    #Z = zs.reshape(X.shape)

    # create colored Z
    colored_z = np.zeros((z.shape[0], z.shape[1], 3))
    for row_num, row in enumerate(z):
        for col_num, val in enumerate(row):
            for color, break_point in zip(colors, break_points):
                if val < break_point:
                    colored_z[row_num, col_num] = color
                    break

    ax.imshow(colored_z, interpolation='none', extent=[xmin, xmax, ymin, ymax])
    # Z = (Z*(k.ynormRange[1]-k.ynormRange[0]))+k.ynormRange[0]

    #Calculate errors
    #zse = np.array([k.predict_var([x,y]) for x,y in zip(np.ravel(X), 
    #    np.ravel(Y))])
    #Ze = zse.reshape(X.shape)

    '''
    spx = (k.X[:,0] * (k.normRange[0][1] - k.normRange[0][0])
            ) + k.normRange[0][0]
    spy = (k.X[:,1] * (k.normRange[1][1] - k.normRange[1][0])
            ) + k.normRange[1][0]
    contour_levels = 25

    CS = ax.contourf(X,Y,Z,contour_levels,zorder=1)
    ax.plot(spx, spy,'ow', zorder=3, alpha=0.1)
    '''
    #ax.set_xlim(xmin, xmax)
    #ax.set_ylim(ymin, ymax)
    #plt.colorbar(ax=ax)
    


    return None # }}}


def create_linear_cmap(name, colors, positions):
    cdict = {}
    for i, primary in zip([0,1,2], ['red', 'green', 'blue']):
        stops = []
        for c, pos in zip(colors, positions):
            new_stop = (pos, c[i], c[i])
            stops.append(new_stop)
        cdict[primary] = stops

    cmap = LinearSegmentedColormap(name, cdict)
    return cmap

def create_stepwise_cmap(name, colors, positions):
    '''
    len(positions) should be len(colors) + 1
    '''
    cdict = {}
    for i, primary in zip([0,1,2], ['red', 'green', 'blue']):
        stops = []
        for j, pos in enumerate(positions):
            color1 = colors[max(0,i-1)]
            color2 = colors[i]
            new_stop = (pos, color1[i], color2[i])
            stops.append(new_stop)
        cdict[primary] = stops

    cmap = LinearSegmentedColormap(name, cdict)
    return cmap

def get_color_for_val(val, colors, break_points):
    for color, break_point in zip(colors, break_points):
        if val < break_point:
            return color
    return colors[-1]

def circle_scatter(ax, x_array, y_array, colors, radius=1, **kwargs):
    for x, y, c in zip(x_array, y_array, colors):
        circle = plt.Circle((x,y), color=c, edgecolor='none', 
                radius=radius, **kwargs)
        ax.add_patch(circle)        
    
def load_soil_data(soil_fpath):
    soil_df = pd.read_csv(soil_fpath)
    soil_cols_to_keep = [
        'fieldId',
        'br_field_ids',
        'ph',
        'cec',
        'k',
        'p',
        's',
        'organicMatter',
        'zn',
        'slope',
        'curve',
        'clay',
        'silt',
        'sand',
        'soil_texture_class',
        'latitude',
        'longitude',
        ]

    soil_df = soil_df[soil_cols_to_keep]

    # Remove records without Field info
    soil_df.dropna(subset=['fieldId'], inplace=True)
    return soil_df
    


if __name__ == '__main__':
    flds = [
        'MNCF',
        'MNWX',
        'MNPX',
        'MNGL',
        'NEAO',
        'ILPO',
        'ILUE',
        'INPO',
        'KSPB',
        #'KSAN'
        ]

    flds = [
        'ILDW',
        'IAEN',
        'Lester',#'MDMO',
        'MNCE',
        'MOMR',
        'MNGL.Glyndon'

        ]
    flds = [
        'ILDW',
        'NDCV',
        'ILPO',#'MDMO',
        'OHOC',
        'KSST',
        'MNGL.Glyndon'

        ]
    
    max_samples = 500000
    nlags = 6
    variogram = 'power'
    variogram = 'linear'
    variogram = 'gaussian'
    variogram = 'exponential'
    grid_resolution = 10
    feat = 'organicMatter'
    feat = 'zn'
    feat = 'p'
    feat = 'k'
    feat = 's'
    feat = 'ph'
    feat = 'cec'
    feat = 'Yield'


    # NOTE: after the soil data is retrieved via the API, put it into a 
    #   DataFrame here to have the code use it. Then replace soil_df below
    #   to have the code use the new data.
    soil_fpath = '2014-2016-locations-samples-nonveris-3.csv'
    soil_df = load_soil_data(soil_fpath)
    yld_df = pd.read_csv('output_BreedingTom.csv')
    yld_df.rename(columns={'Location':'fieldId'}, inplace=True)

    
    

    if True:
        make_fig(yld_df, flds=flds, feat=feat, show_fig=True, nlags=nlags, 
                max_samples=max_samples,
                variogram=variogram, grid_resolution=grid_resolution)
    else:
        make_fig_multi_core()

#soil_df.groupby(['fieldId'])['Yield'].mean()
