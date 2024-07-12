from config import *
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