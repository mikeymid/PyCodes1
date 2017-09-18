import pandas as pd

from IPython import embed

def load_data(soil_file_path, yield_file_path, max_n_samples_per_field=500,
    min_n_samples_per_field=0):

    
    soil_df = pd.read_csv(soil_file_path)
    yield_df = pd.read_csv(yield_file_path)

    yield_cols_to_keep = [
        'GERMPLASM_ID',
        'LOCID',
        'LOCATION',
        'EXPER_STAGE_REF_ID',
        'NUM_VALUE',
        'PLOT_ID',
        'UNIT']

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
    
    # Make sure there are fewer than 500 samples for each field
    n_samps_by_field = soil_df.fieldId.value_counts()
    nonveris_fields = list(n_samps_by_field[n_samps_by_field.between(
            min_n_samples_per_field, max_n_samples_per_field)].index)

    yield_df = yield_df[yield_df.LOCID.isin(nonveris_fields)]

    return yield_df, soil_df

def create_field_sample_counts_file(soil_file_path, yield_file_path, 
        max_n_samples_per_field=500, min_n_samples_per_field=0):
    soil_df = pd.read_csv(soil_file_path)
    yield_df = pd.read_csv(yield_file_path)

    n_samps_by_field = soil_df.fieldId.value_counts()
    n_samps_by_field = n_samps_by_field[n_samps_by_field.between(
        min_n_samples_per_field, max_n_samples_per_field)]

    df = pd.DataFrame()
    df['field'] = n_samps_by_field.index
    df['n_soil_samples'] = n_samps_by_field.values

    # add a column to indicate if we have yield data
    yld_flds = list(yield_df.LOCID.unique())
    df['has_yield_data'] = df.field.isin(yld_flds)
    
    # add a row number
    df['row_num'] = df.index + 1

    df.to_csv('Soil_sample_counts.csv', index=False)


if __name__ == '__main__':
    yield_fpath = 'latest_2016cornyld.csv'
    soil_fpath = 'samples_3.csv'
    soil_fpath = '2014-2016-locations-samples-nonveris-3.csv'
    #load_data(soil_fpath, yield_fpath)
    create_field_sample_counts_file(soil_fpath, yield_fpath)
