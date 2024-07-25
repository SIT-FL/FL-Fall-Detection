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
def preprocess_dataset(df_dataset, binary_multi='binary'):
    # Convert multi-dimensional arrays to numpy arrays
    for column in feature_columns:
        df_dataset[column] = df_dataset[column].apply(lambda x: np.array([i[0] for i in x]))
        # Truncate sequences to the minimum length (for simplicity)
        sequence_length = min(len(seq) for seq in df_dataset['w'])
        # print(f"[{column}]: Truncating sequences to length: {sequence_length}")
        df_dataset[column] = df_dataset[column].apply(lambda x: x[:sequence_length])

    # Extract features and labels
    features = df_dataset[feature_columns].values
    features = np.array(features.tolist(), dtype=np.float32)

    # Convert to PyTorch tensors
    subjects =df_dataset['subject'].values
    inputs = torch.tensor(features, dtype=torch.float32)
    inputs = inputs.permute(0, 2, 1)

    # Convert scenario_encoded to labels (targets)
    if binary_multi == 'both':
        labels_binary = df_dataset['class_encoded'].values.astype(np.int64)
        labels_multi = df_dataset['scenario_encoded'].values.astype(np.int64)
        targets_binary = torch.tensor(labels_binary, dtype=torch.long)
        targets_multi = torch.tensor(labels_multi, dtype=torch.long)
        return subjects, inputs, torch.concatenate((targets_binary[:, None], targets_multi[:, None]), 1)
    else:
        if binary_multi == 'binary':
            label_column = 'class_encoded'
        else:
            label_column = 'scenario_encoded'
        labels = df_dataset[label_column].values.astype(np.int64)

        targets = torch.tensor(labels, dtype=torch.long)        
        return subjects, inputs, targets

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

def train_local_model_cascade(model, dataloader, criterion, optimizer, device):
    model.train()
    for inputs, targets in dataloader:
        targets_binary, targets_multi = targets[:, 0], targets[:, 1]
        inputs, targets_binary, targets_multi = inputs.to(device), targets_binary.to(device), targets_multi.to(device)
        optimizer.zero_grad()
        binary_out, multi_out = model(inputs)
        binary_loss = criterion(binary_out, targets_binary)
        multi_loss = criterion(multi_out, targets_multi) 
        loss = binary_loss + multi_loss
        loss.backward()
        optimizer.step()

    return model 

def train_local_model_fedprox_cascade(global_model, local_model, dataloader, criterion, optimizer, device, mu):
    local_model.train()
    global_model.to(device)
    for inputs, targets in dataloader:
        targets_binary, targets_multi = targets[:, 0], targets[:, 1]
        inputs, targets_binary, targets_multi = inputs.to(device), targets_binary.to(device), targets_multi.to(device)
        optimizer.zero_grad()
        binary_out, multi_out = local_model(inputs)
        binary_loss = criterion(binary_out, targets_binary)
        multi_loss = criterion(multi_out, targets_multi) 
        loss = binary_loss + multi_loss

        # Proximal term
        proximal_term = 0.0
        for w, w_t in zip(local_model.parameters(), global_model.parameters()):
            proximal_term += (w - w_t).norm(2)

        loss += (mu / 2) * proximal_term
        loss.backward()
        optimizer.step()
    return local_model

def train_local_model_hierarchical(model, dataloader, criterion, optimizer, device):
    for k in model.keys():
        model[k].train()

    for inputs, targets in dataloader:
        targets_binary, targets_multi = targets[:, 0], targets[:, 1]
        inputs, targets_binary, targets_multi = inputs.to(device), targets_binary.to(device), targets_multi.to(device)
        for k in optimizer.keys():
            optimizer[k].zero_grad()

        binary_out, (hn, cn) = model['binary_lstm'](inputs)
        binary_loss = criterion(binary_out, targets_binary)
        binary_loss.backward(retain_graph=True)
        optimizer['binary_lstm'].step()

        # train the fall model 
        targets_fall = targets_multi[targets_binary == 0]
        inputs_fall = inputs[targets_binary == 0]
        if inputs_fall.size(0) > 0:           
            # h = hn.clone()[:, targets_binary == 0, :]
            # c = cn.clone()[:, targets_binary == 0, :]
            fall_out, _ = model['fall_lstm'](inputs_fall)
            fall_loss = criterion(fall_out, targets_fall)
            fall_loss.backward()
            optimizer['fall_lstm'].step()

        # train the non-fall model
        targets_non_fall = targets_multi[targets_binary == 1]
        inputs_non_fall = inputs[targets_binary == 1]
        if inputs_non_fall.size(0) > 0: 
            # h = hn.clone()[:, targets_binary == 1, :]
            # c = cn.clone()[:, targets_binary == 1, :]
            non_fall_out, _ = model['non_fall_lstm'](inputs_non_fall)
            non_fall_loss = criterion(non_fall_out, targets_non_fall)
            non_fall_loss.backward()
            optimizer['non_fall_lstm'].step()

    return model

def train_local_model_fedprox_hierarchical(global_model, local_model, dataloader, criterion, optimizer, device, mu):
    for k in local_model.keys():
        local_model[k].train()
        global_model[k].to(device)
        
    for inputs, targets in dataloader:
        targets_binary, targets_multi = targets[:, 0], targets[:, 1]
        inputs, targets_binary, targets_multi = inputs.to(device), targets_binary.to(device), targets_multi.to(device)
        for k in optimizer.keys():
            optimizer[k].zero_grad()

        binary_out, (hn, cn) = local_model['binary_lstm'](inputs)
        binary_loss = criterion(binary_out, targets_binary)
        optimizer['binary_lstm'].step()

        # Proximal term
        proximal_term = 0.0
        for w, w_t in zip(local_model['binary_lstm'].parameters(), global_model['binary_lstm'].parameters()):
            proximal_term += (w - w_t).norm(2)

        binary_loss += (mu / 2) * proximal_term
        binary_loss.backward()
        optimizer['binary_lstm'].step()

        # train the fall model 
        targets_fall = targets_multi[targets_binary == 0]
        inputs_fall = inputs[targets_binary == 0]
        if inputs_fall.size(0) > 0:           
            fall_out, _ = local_model['fall_lstm'](inputs_fall)
            fall_loss = criterion(fall_out, targets_fall)
            optimizer['fall_lstm'].step()

            # Proximal term
            proximal_term = 0.0
            for w, w_t in zip(local_model['fall_lstm'].parameters(), global_model['fall_lstm'].parameters()):
                proximal_term += (w - w_t).norm(2)

            fall_loss += (mu / 2) * proximal_term
            fall_loss.backward()
            optimizer['fall_lstm'].step()

        # train the non-fall model
        targets_non_fall = targets_multi[targets_binary == 1]
        inputs_non_fall = inputs[targets_binary == 1]
        if inputs_non_fall.size(0) > 0: 
            non_fall_out, _ = local_model['non_fall_lstm'](inputs_non_fall)
            non_fall_loss = criterion(non_fall_out, targets_non_fall)
            optimizer['non_fall_lstm'].step()

            # Proximal term
            proximal_term = 0.0
            for w, w_t in zip(local_model['non_fall_lstm'].parameters(), global_model['non_fall_lstm'].parameters()):
                proximal_term += (w - w_t).norm(2)

            non_fall_loss += (mu / 2) * proximal_term
            non_fall_loss.backward()
            optimizer['non_fall_lstm'].step()
    return local_model

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
    return local_model

# Function to perform simple federated averaging
def federated_averaging(global_model, local_models):
    global_dict = global_model.state_dict()
    for key in global_dict.keys():
        global_dict[key] = torch.stack([local_models[i].state_dict()[key].float() for i in range(len(local_models))], 0).mean(0)
    global_model.load_state_dict(global_dict)

def federated_averaging_hierarchical(global_model, local_models):
    for k in global_model.keys():
        global_dict = global_model[k].state_dict()
        for key in global_dict.keys():
            global_dict[key] = torch.stack([local_models[i][k].state_dict()[key].float() for i in range(len(local_models))], 0).mean(0)
        global_model[k].load_state_dict(global_dict)
    
# Function to return the performance for the model
def get_metrics(list_y, list_pred, class_names, save_path, binary_multi='binary'):
    # filter class_names dict based on the unique values in list_y
    class_names = np.asarray(list(class_names))[np.unique(np.concatenate([list_y, list_pred]))]

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
    cm = confusion_matrix(list_y, list_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    if binary_multi == 'binary':
        plt.figure(figsize=(12, 10))  # Adjust the size as per your preference
        disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=class_names)
        disp.plot(cmap=plt.cm.Blues, values_format='.4f', xticks_rotation='vertical')  # Rotates x-axis labels vertically
    else:
        plt.figure(figsize=(30, 10))  # Adjust the size as per your preference
        disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=class_names)
        disp.plot(cmap=plt.cm.Blues, values_format='.2f', xticks_rotation='vertical')

        for labels in disp.text_.ravel():
            labels.set_fontsize(8)  
    plt.title('Confusion Matrix')
    plt.tight_layout()  # Ensures labels are not cut off
    plt.savefig(save_path, bbox_inches='tight')
    
    return results
    
# Function to evaluate a model
def evaluate_model(model, dataloader, criterion, device, name, save_path, binary_multi='binary'):
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

    if binary_multi == 'binary':
        mapping = class_mapping
    else:
        mapping = scenario_mapping
    
    return get_metrics(all_targets, all_predictions, mapping.values(), f"{save_path}/{name}_confusion_matrix.png", binary_multi)

def evaluate_cascade_model(model, dataloader, criterion, device, name, save_path):
    # Evaluation mode
    model.eval()

    # Lists to store predictions and true labels
    all_binary_predictions = []
    all_binary_targets = []
    all_multi_predictions = []
    all_multi_targets = []
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            targets_binary, targets_multi = targets[:, 0], targets[:, 1]
            inputs, targets_binary, targets_multi = inputs.to(device), targets_binary.to(device), targets_multi.to(device)
            binary_out, multi_out = model(inputs)

            _, binary_predicted = torch.max(binary_out, 1)  # Get the predicted class indices
            all_binary_predictions.extend(binary_predicted.cpu().numpy())
            all_binary_targets.extend(targets_binary.cpu().numpy())

            _, multi_predicted = torch.max(multi_out, 1)  # Get the predicted class indices
            all_multi_predictions.extend(multi_predicted.cpu().numpy())
            all_multi_targets.extend(targets_multi.cpu().numpy())

    # Convert lists to numpy arrays
    all_binary_predictions = np.array(all_binary_predictions)
    all_binary_targets = np.array(all_binary_targets)
    all_multi_predictions = np.array(all_multi_predictions)
    all_multi_targets = np.array(all_multi_targets)

    results_binary = get_metrics(all_binary_targets, all_binary_predictions, class_mapping.values(), f"{save_path}/{name}_binary_confusion_matrix.png", 'binary')
    results_multi = get_metrics(all_multi_targets, all_multi_predictions, scenario_mapping.values(), f"{save_path}/{name}_multi_confusion_matrix.png", 'multi')
    
    return results_binary, results_multi


def evaluate_hierarchical_model(model, dataloader, criterion, device, name, save_path):
    # Evaluation mode
    for k in model.keys():
        model[k].eval()

    # Lists to store predictions and true labels
    all_binary_predictions = []
    all_binary_targets = []
    all_multi_predictions = []
    all_multi_targets = []
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            targets_binary, targets_multi = targets[:, 0], targets[:, 1]
            inputs, targets_binary, targets_multi = inputs.to(device), targets_binary.to(device), targets_multi.to(device)
            binary_out, _ = model['binary_lstm'](inputs)


            _, binary_predicted = torch.max(binary_out, 1)  # Get the predicted class indices
            all_binary_predictions.extend(binary_predicted.cpu().numpy())
            all_binary_targets.extend(targets_binary.cpu().numpy())

            temp_multi_predictions = []
            for i, (inp, binary_pred) in enumerate(zip(inputs, binary_predicted)):
                if binary_pred == 0:
                    multi_out, _ = model['fall_lstm'](inp[None, ...])
                    _, multi_predicted = torch.max(multi_out, 1)
                    temp_multi_predictions.extend(multi_predicted.cpu().numpy())
                else:
                    multi_out, _ = model['non_fall_lstm'](inp[None, ...])
                    _, multi_predicted = torch.max(multi_out, 1)
                    temp_multi_predictions.extend(multi_predicted.cpu().numpy())

            all_multi_predictions.extend(temp_multi_predictions)
            all_multi_targets.extend(targets_multi.cpu().numpy())

    # Convert lists to numpy arrays
    all_binary_predictions = np.array(all_binary_predictions)
    all_binary_targets = np.array(all_binary_targets)
    all_multi_predictions = np.array(all_multi_predictions)
    all_multi_targets = np.array(all_multi_targets)

    results_binary = get_metrics(all_binary_targets, all_binary_predictions, class_mapping.values(), f"{save_path}/{name}_binary_confusion_matrix.png", 'binary')
    results_multi = get_metrics(all_multi_targets, all_multi_predictions, scenario_mapping.values(), f"{save_path}/{name}_multi_confusion_matrix.png", 'multi')
    
    return results_binary, results_multi
