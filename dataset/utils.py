from config import *
from datetime import datetime
import pandas as pd
import pickle

# Function to pickle an object
def pickle_object (pickle_object, filepath):
    file_pickle = open (filepath, 'wb')
    pickle.dump (pickle_object, file_pickle)
    file_pickle.close ()

# Function to load pickled object
def load_pickle (filepath):
    file_pickle = open (filepath, 'rb')
    pickled_object = pickle.load (file_pickle)
    file_pickle.close ()
    return pickled_object

# Function to expand given dataframe for each sensor reading
def expand_sensor_data(df, list_columns):
    # Explode the DataFrame based on the list columns
    df_new = pd.concat([df.drop(columns=list_columns), df[list_columns].apply(pd.Series.explode)], axis=1)
    # Convert single element list to their values
    df_new[list_columns] = df_new[list_columns].map(lambda x: x[0] if isinstance(x, list) and len(x) == 1 else x)
    # Convert list of time values to a timestamp object
    df_new['time'] = df_new['time'].apply(lambda x: datetime(year=int(x[0]), month=int(x[1]), day=int(x[2]), hour=int(x[3]), minute=int(x[4]), second=int(x[5]), microsecond=int((x[5] % 1) * 1e6)))
    df_new.reset_index(inplace=True, drop=True)
    return df_new