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

# Function to train a local model
def train_local_model(model, dataloader, criterion, optimizer, device):
    model.train()
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

# Functino to train a local model with FedProx
def train_local_model_fedprox(global_model, local_model, dataloader, criterion, optimizer, device, mu):
    local_model.train()
    global_model.to(device)
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = local_model(inputs)
        loss = criterion(outputs, targets)

        # Proximal term
        proximal_term = 0.0
        for w, w_t in zip(local_model.parameters(), global_model.parameters()):
            proximal_term += (w - w_t).norm(2)

        loss += (mu / 2) * proximal_term
        loss.backward()
        optimizer.step()

# Function to perform simple federated averaging
def federated_averaging(global_model, local_models):
    global_dict = global_model.state_dict()
    for key in global_dict.keys():
        global_dict[key] = torch.stack([local_models[i].state_dict()[key].float() for i in range(len(local_models))], 0).mean(0)
    global_model.load_state_dict(global_dict)
    
# Function to return the performance for the model
def get_metrics(list_y, list_pred, class_names, save_path):
    # Obtain metrics
    results = {
        "accuracy": accuracy_score(list_y, list_pred),
        "precision": precision_score(list_y, list_pred, average='weighted', zero_division=0),
        "recall": recall_score(list_y, list_pred, average='weighted', zero_division=0),
        "f1": f1_score(list_y, list_pred, average='weighted', zero_division=0),
        "mcc": matthews_corrcoef(list_y, list_pred),
        "kappa": cohen_kappa_score(list_y, list_pred),
        "hamming_loss_val": hamming_loss(list_y, list_pred),
        "class_report": classification_report(list_y, list_pred, target_names=class_names),
    }
    # Confusion Matrix
    plt.figure(figsize=(12, 10))  # Adjust the size as per your preference
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(list_y, list_pred), display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, values_format='.0f', xticks_rotation='vertical')  # Rotates x-axis labels vertically
    plt.title('Confusion Matrix')
    plt.tight_layout()  # Ensures labels are not cut off
    plt.savefig(save_path, bbox_inches='tight')
    
    return results
    
# Function to evaluate a model
def evaluate_model(model, dataloader, criterion, device, name, save_path):
    # Evaluation mode
    model.eval()

    # Lists to store predictions and true labels
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)  # Get the predicted class indices
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
    # Convert lists to numpy arrays
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    
    return get_metrics(all_targets, all_predictions, scenario_mapping.values(), f"{save_path}/{name}_confusion_matrix.png")

