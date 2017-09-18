import pandas as pd
from IPython import embed

def get_summary_df(df, depth_in=36, cols_to_keep=[]):
    '''
    Reduce the hundreds of columns to just a few, representing basic soil 
    features for the horizons near the surface.
    
    '''
    cols_to_keep = cols_to_keep + [
            #'ID',
            #'mukey',
            #'centLat',
            #'centLong',
            'major_cmp_pct',
            'AWS',
            'organic_matter',
            'sand',
            'clay',
            'cec',
            'ph',
            ]

    property_names = {
            'organic_matter':'om_rep',
            'sand':'sandtotal_rep',
            'clay':'claytotal_rep',
            'cec':'cec7_rep',
            'ph':'ph1to1h2o_rep'
            }

    # Get horizon numbers
    horizon_cols = [c for c in df.columns if 'componentHorizon' in c]
    horizon_nums = list(set([c.split('componentHorizon')[1].split('_')[0] 
            for c in horizon_cols]))


    # add horizon weights
    max_depth_in = depth_in
    max_depth_cm = max_depth_in * 2.54
    df.fillna(0, inplace=True)
    for h_num in horizon_nums:
        df['weight'+h_num] = df.apply(get_horizon_weight,
                args=(h_num, max_depth_cm), axis=1)
        
    # add the average values
    for feat in property_names.keys():
        old_label = property_names[feat]
        df[feat] = df.apply(get_weighted_avg, args=(old_label, horizon_nums), 
                axis=1)

    
    df.rename(columns={'component_comppct_rep':'major_cmp_pct',
            'mapUnit_mukey':'mukey',
            'mapUnitAggregate_aws0100wta':'AWS'}, inplace=True)
    df = df[cols_to_keep]

    return df
    

def get_horizon_weight(row, horizon_num, max_depth_cm):
    bottom_depth = row['componentHorizon' + horizon_num + '_hzdepb_rep'] 
    top_depth = row['componentHorizon' + horizon_num + '_hzdept_rep'] 
    weight = max(min(bottom_depth, max_depth_cm) - top_depth, 0)/float(
            max_depth_cm)
    return weight

def get_weighted_avg(row, old_label, horizon_nums):
    value_sum = 0
    weights_sum = 0
    for h_num in horizon_nums:
        weight = row['weight'+h_num]
        if weight == 0:
            continue
        col_name = 'componentHorizon' + h_num + '_' + old_label 
        try:
            value_sum += row[col_name]
            weights_sum += weight
        except:
            print(col_name + ' was not returned by the API. An average will ' +
                    'be calculated without the data from that horizon.')

    try:
        weighted_avg = value_sum / float(weights_sum)
    except ZeroDivisionError:
        print('A soil region was found with no data for ' + old_label + 
                '. A null value will be returned.')
        return None
        
    return weighted_avg
