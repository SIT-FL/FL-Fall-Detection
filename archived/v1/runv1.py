from utilsv1 import *
import torch.optim as optim
import argparse, os, json

# Simple LSTM model for experimental purposes
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def main(args):
    # Load datasets
    df_train = load_pickle(TRAIN_DATASET_FILE)
    df_val = load_pickle(VAL_DATASET_FILE)

    # Prepare tensors and datasets for training and validation
    train_inputs, train_targets = preprocess_dataset(df_train)
    val_inputs, val_targets = preprocess_dataset(df_val)
    train_dataset = SensorDataset(train_inputs, train_targets)
    val_dataset = SensorDataset(val_inputs, val_targets)

    # Create DataLoader objects
    batch_size = 64
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Hyperparameters (pre-defined for experiment)
    input_size = train_inputs.shape[2]
    num_classes = len(np.unique(train_targets))
    hidden_size = 128
    num_layers = 2
    learning_rate = 0.01
    num_epochs = 10 # Reduced local epochs
    num_clients = 10
    num_rounds = 10
    mu = 0.01  # FedProx hyperparameter

    # Initialize global model, criterion, and optimizer
    global_model = LSTMModel(input_size, hidden_size, num_layers, num_classes)
    criterion = nn.CrossEntropyLoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    global_model.to(device)

    # Split the data among clients
    client_data_size = len(train_dataset) // num_clients
    client_datasets = [torch.utils.data.Subset(train_dataset, range(i*client_data_size, (i+1)*client_data_size)) for i in range(num_clients)]
    client_dataloaders = [DataLoader(client_dataset, batch_size=batch_size, shuffle=True) for client_dataset in client_datasets]

    # Create folder for results
    save_path = f"{SAVE_FOLDER}/{datetime.now().strftime('%d-%m-%Y-%H%M%S')}"
    os.makedirs(save_path)
    
    overall_results = {}
    
    # Federated learning rounds
    for round in range(num_rounds):
        round_results = {}
        print(f"~ Federated Learning Round {round+1}/{num_rounds} ~")
        # Initialise list of individual local models
        local_models = []
        for client in range(num_clients):
            # Create a new model instance and load the global model state dict
            local_model = LSTMModel(input_size, hidden_size, num_layers, num_classes).to(device)
            local_model.load_state_dict(global_model.state_dict())
            optimizer = optim.Adam(local_model.parameters(), lr=learning_rate)

            # Train the local model (normal or FedProx)
            if args.fedprox:
                train_local_model_fedprox(global_model, local_model, client_dataloaders[client], criterion, optimizer, device, mu)
            else:
                train_local_model(local_model, client_dataloaders[client], criterion, optimizer, device)

            local_models.append(local_model)
            
            results = evaluate_model(local_model, val_dataloader, criterion, device, f"round_{round + 1}_local_{client + 1}", save_path)
            round_results[f"client_{client + 1}"] = results
            
            print(f"Local Model {client + 1}:\nAccuracy: {results['accuracy']}\nPrecision: {results['precision']}\nRecall: {results['recall']}\nF1 Score: {results['f1']}", end="\n\n")

        # Perform federated averaging to update the global model
        federated_averaging(global_model, local_models)
        
        # Evaluate global model
        results = evaluate_model(global_model, val_dataloader, criterion, device, f"round_{round + 1}_global", save_path)
        round_results["global"] = results
        print(f"Global Model:\nAccuracy: {results['accuracy']}\nPrecision: {results['precision']}\nRecall: {results['recall']}\nF1 Score: {results['f1']}", end="\n\n")
        
        overall_results[f"round_{round + 1}"] = round_results

    # Export results
    with open(f"{save_path}/results.json", 'w') as json_file:
        json.dump(overall_results, json_file, indent=4)
        
# Program entrypoint
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Federated Learning with FedProx')
    parser.add_argument('--fedprox', action='store_true', help='Use FedProx algorithm')
    args = parser.parse_args()
    main(args)
