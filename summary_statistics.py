import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pyKriging.krige import kriging

import location_selection
from IPython import embed

def make_fig_for_fld(fld_df, save_as=None):

    # --------- Visual settings ----------------
    l_font_size = 20
    s_font_size = 16


    # --------- Layout Settings ----------------
    top_padding = 0.05
    bottom_padding = 0.02
    left_padding = 0.02
    right_padding = 0.02
    feat_vspace = 0.08
    feat_hspace = 0.04
    minor_vspace = 0.02
    minor_hspace = 0.02

    fig_size_in = (24,14)

    num_rows = 2
    num_cols = 4

    num_subplots_per_fig = num_rows * num_cols

    ax_height = (1 - (num_rows-1)*feat_vspace -num_rows*minor_vspace - 
            top_padding - bottom_padding)/ float(num_rows*2)
    krig_height = ax_height
    hist_height = ax_height
    krig_width = (1 - left_padding - right_padding - (num_cols-1)*feat_hspace
            ) / float(num_cols)
    hist_width = krig_width / 2.0

    potential_features = [
        'ph',
        'cec',
        'k',
        'p',
        's',
        'n',
        'organicMatter',
        'zn',
        ]
    features = [f for f in potential_features if f in fld_df.columns]


    X = fld_df[['latitude', 'longitude']].values

    figs = {}
        
    for i, feat in enumerate(features):
        fig_num = i // num_subplots_per_fig + 1
        subplot_num = i % num_subplots_per_fig
        print(feat)
        if subplot_num == 0: #this is the first feat for the figure
            fig = plt.figure()
            figs[fig_num] = fig

        row_num = subplot_num // num_cols
        col_num = subplot_num % num_cols
        
        # create the axes
        krig_left = left_padding + col_num*(krig_width+feat_hspace)
        krig_top = (1 - top_padding - row_num *
                (krig_height + hist_height + minor_vspace + feat_vspace))
        krig_bottom = krig_top - krig_height
        krig_rect = (krig_left, krig_bottom, krig_width, krig_height)
        krig_ax = fig.add_axes(krig_rect)

        hist_left = krig_left
        hist_bottom = krig_bottom - minor_vspace - hist_height

        hist_rect = (hist_left, hist_bottom, hist_width, hist_height)
        hist_ax = fig.add_axes(hist_rect)

        # do Kriging
        y = fld_df[feat].values
        do_and_plot_kriging(X, y, krig_ax)



        #krig_ax.plot([1,2,3,4,5,6],[1,3,2,4,5,6])
        hist_ax.hist(fld_df[feat].values, bins=10)

        # add text descriptions
        mean_text = 'Mean: ' + str(round(fld_df[feat].mean(),3))
        std_text = 'Std: ' + str(round(fld_df[feat].std(),3))
        min_text = 'Min: ' + str(round(fld_df[feat].min(),3))
        max_text = 'Max: ' + str(round(fld_df[feat].max(),3))
        desciption = (mean_text + '\n' + std_text + '\n' + min_text + '\n' + 
                max_text)

        hist_ax.text(1.2, 0.8, desciption, transform=hist_ax.transAxes,
                verticalalignment='top', horizontalalignment='left', 
                fontsize=l_font_size)

        # Add labels
        krig_ax.set_title(feat, fontsize=l_font_size)

        

    if save_as is None:
        plt.show()
    else:
        for fig_num in figs.keys():
            fig = figs[fig_num]
            fig.set_size_inches(fig_size_in)
            fig.savefig(save_as + '_' + str(fig_num) + '.png', dpi=300)


def do_and_plot_kriging(X, y, ax):
    xmin, ymin = X.min(axis=0)
    xmax, ymax = X.max(axis=0)
    k = kriging(X, y)
    k.train()

    numiter = 2 
    for i in range(numiter):  
        print('Infill iteration {0} of {1}....'.format(i + 1, numiter))
        newpoints = k.infill(1)
        for point in newpoints:
            k.addPoint(point, k.predict(point))
        k.train()

    samplePoints = list(zip(*k.X))

    # Create a set of data to plot
    plotgrid = 61
    x = np.linspace(k.normRange[0][0], k.normRange[0][1], num=plotgrid)
    y = np.linspace(k.normRange[1][0], k.normRange[1][1], num=plotgrid)

    # x = np.linspace(0, 1, num=plotgrid)
    # y = np.linspace(0, 1, num=plotgrid)
    X, Y = np.meshgrid(x, y)

    # Predict based on the optimized results

    zs = np.array([k.predict([x,y]) for x,y in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(X.shape)
    # Z = (Z*(k.ynormRange[1]-k.ynormRange[0]))+k.ynormRange[0]

    #Calculate errors
    #zse = np.array([k.predict_var([x,y]) for x,y in zip(np.ravel(X), 
    #    np.ravel(Y))])
    #Ze = zse.reshape(X.shape)

    spx = (k.X[:,0] * (k.normRange[0][1] - k.normRange[0][0])
            ) + k.normRange[0][0]
    spy = (k.X[:,1] * (k.normRange[1][1] - k.normRange[1][0])
            ) + k.normRange[1][0]
    contour_levels = 25

    CS = ax.contourf(X,Y,Z,contour_levels,zorder=1)
    ax.plot(spx, spy,'ow', zorder=3, alpha=0.1)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    #plt.colorbar(ax=ax)

        

if __name__ == '__main__':
    fld = 'MNPX'
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
        'KSAN'
        ]

    yield_fpath = 'latest_2016cornyld.csv'
    soil_fpath = '2014-2016-locations-samples-nonveris-3.csv'

    yield_df, soil_df = location_selection.load_data(soil_fpath, yield_fpath)
    for fld in flds:
        print(fld)
        fld_df = soil_df[soil_df.fieldId == fld]


        make_fig_for_fld(fld_df, save_as=fld)
