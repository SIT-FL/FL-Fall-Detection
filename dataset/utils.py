import pickle, torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from config import *
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score, classification_report, matthews_corrcoef, cohen_kappa_score, hamming_loss

""" Classes """
# Custom Dataset class for fall-detection sensor data
class SensorDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

""" Functions """
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

# Function to preprocess dataset
def preprocess_dataset(df_dataset):
    # Convert multi-dimensional arrays to numpy arrays
    for column in feature_columns:
        df_dataset[column] = df_dataset[column].apply(lambda x: np.array([i[0] for i in x]))
        # Truncate sequences to the minimum length (for simplicity)
        sequence_length = min(len(seq) for seq in df_dataset['w'])
        df_dataset[column] = df_dataset[column].apply(lambda x: x[:sequence_length])

    # Extract features and labels
    features = df_dataset[feature_columns].values
    features = np.array(features.tolist(), dtype=np.float32)

    # Convert scenario_encoded to labels (targets)
    labels = df_dataset[label_column].values.astype(np.int64)

    # Convert to PyTorch tensors
    inputs = torch.tensor(features, dtype=torch.float32)
    inputs = inputs.permute(0, 2, 1)
    targets = torch.tensor(labels, dtype=torch.long)
    
    return inputs, targets

# Function to return the performance for the model
def get_metrics(list_y, list_pred, class_names, print_results=False):
    # Obtain metrics
    results = {
        "accuracy": accuracy_score(list_y, list_pred),
        "precision": precision_score(list_y, list_pred, average='weighted', zero_division=0),
        "recall": recall_score(list_y, list_pred, average='weighted', zero_division=0),
        "f1": f1_score(list_y, list_pred, average='weighted', zero_division=0),
        "mcc": matthews_corrcoef(list_y, list_pred),
        "kappa": cohen_kappa_score(list_y, list_pred),
        "hamming_loss_val": hamming_loss(list_y, list_pred),
        "cm": confusion_matrix(list_y, list_pred),
        "class_report": classification_report(list_y, list_pred, target_names=class_names),
    }
    if(print_results):
        print("Accuracy:", results['accuracy'])                                    # Model Accuracy: How often is the classifier correct
        print("Precision:", results['precision'])                                  # Model Precision: what percentage of positive tuples are labeled as such?
        print("Recall:", results['recall'])                                        # Model Recall: what percentage of positive tuples are labelled as such?
        print("F1 Score:", results['f1'])                                          # F1 Score: The weighted average of Precision and Recall
        print("Matthews Correlation Coefficient (MCC):", results['mcc'])           # Matthews Correlation Coefficient (MCC): Measures the quality of classifications
        print("Cohen's Kappa:", results['kappa'])                                  # Cohen's Kappa: Measures inter-rater agreement for categorical items    
        print("Hamming Loss:", results['hamming_loss_val'], end='\n\n')            # Hamming Loss: The fraction of labels that are incorrectly predicted
        print("Confusion Matrix:\n", results['cm'], end="\n\n")

        # Confusion Matrix
        plt.figure(figsize=(12, 10))  # Adjust the size as per your preference
        disp = ConfusionMatrixDisplay(confusion_matrix=results['cm'], display_labels=class_names)
        disp.plot(cmap=plt.cm.Blues, values_format='.0f', xticks_rotation='vertical')  # Rotates x-axis labels vertically
        plt.title('Confusion Matrix')
        plt.tight_layout()  # Ensures labels are not cut off
        plt.show()

        print("Classification Report:\n", results['class_report'], end="\n\n")
    return results