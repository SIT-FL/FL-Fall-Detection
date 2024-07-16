""" Configurations """
# Parameters
RAND_SEED = 123456
SPLIT_SIZE = 0.2

# Variables
feature_columns = ['w', 'x', 'y', 'z', 'droll', 'dpitch', 'dyaw', 'ax', 'ay', 'az', 'heart']
label_column = 'scenario_encoded'
scenario_mapping = {0: 'bed', 1: 'chair', 2: 'clap', 3: 'cloth', 4: 'eat', 5: 'fall1', 6: 'fall2', 
                    7: 'fall3', 8: 'fall4', 9: 'fall5', 10: 'fall6', 11: 'hair', 12: 'shoe', 
                    13: 'stair', 14: 'teeth', 15: 'walk', 16: 'wash', 17: 'write', 18: 'zip'}
class_mapping = {0: 'fall', 1: 'non-fall'}

# Files and folders
DATASET_FOLDER = "./dataset"
SAVE_FOLDER = "./results"
EXPORT_FOLDER = "./dataset/export"
DATASET_FILE = f"{EXPORT_FOLDER}/dataset.pkl"
TRAIN_DATASET_FILE = f"{EXPORT_FOLDER}/train_dataset.pkl"
VAL_DATASET_FILE = f"{EXPORT_FOLDER}/val_dataset.pkl"