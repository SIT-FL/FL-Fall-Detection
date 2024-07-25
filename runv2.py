from utilsv1 import *
import torch.optim as optim
import argparse, os, json
import warnings
warnings.filterwarnings("ignore")


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

    clients_train = df_train['subject'].unique()
    clients_val = df_val['subject'].unique()
    clients_train_count = df_train['subject'].value_counts().to_dict()
    clients_val_count = df_val['subject'].value_counts().to_dict()

    # Prepare tensors and datasets for training and validation
    train_subjects, train_inputs, train_targets = preprocess_dataset(df_train, binary_multi=args.binary_multi)
    val_subjects, val_inputs, val_targets = preprocess_dataset(df_val, binary_multi=args.binary_multi)

    if args.binary_multi == "multi":
        # if multi-class, it seems that each subject only have one set of data for each class
        # in this case, we do not have extra data for local testing
        # note: the local training in this case is a one-shot learning
        # we can evaluate the global model performance on the seen data of seen subjects (which is not a good evaluation design)
        # in this case, we can focus the evaluation on the global model on unseen subjects
        client_train_datasets = [SensorDataset(train_inputs[train_subjects == sub], train_targets[train_subjects == sub]) for sub in clients_train]
        client_test_datasets = client_train_datasets #unfortunately
    else:
        # Create datasets per subject
        # for each subject: split the data into train and test
        # obj: this is to verify the client performance on client's own unseen data
        client_train_datasets, client_test_datasets = [], []
        for sub in clients_train:
            temp_inputs = train_inputs[train_subjects == sub]
            temp_targets = train_targets[train_subjects == sub]

            # need to make sure the test set has all the labels
            for i, t in enumerate(np.unique(temp_targets)):
                temp_inputs_t = temp_inputs[temp_targets == t]
                temp_targets_t = temp_targets[temp_targets == t]

                temp_inputs_t_size = len(temp_inputs_t)
                temp_inputs_t_train = temp_inputs_t[:int(temp_inputs_t_size*(1-SPLIT_SIZE))]
                temp_targets_t_train = temp_targets_t[:int(temp_inputs_t_size*(1-SPLIT_SIZE))]
                temp_inputs_t_test = temp_inputs_t[int(temp_inputs_t_size*(1-SPLIT_SIZE)):]
                temp_targets_t_test = temp_targets_t[int(temp_inputs_t_size*(1-SPLIT_SIZE)):]

                if i == 0:
                    client_train_input = temp_inputs_t_train
                    client_train_target = temp_targets_t_train
                    client_test_input = temp_inputs_t_test
                    client_test_target = temp_targets_t_test
                else:
                    client_train_input = np.concatenate((client_train_input, temp_inputs_t_train))
                    client_train_target = np.concatenate((client_train_target, temp_targets_t_train))
                    client_test_input = np.concatenate((client_test_input, temp_inputs_t_test))
                    client_test_target = np.concatenate((client_test_target, temp_targets_t_test))

            client_train_datasets.append(SensorDataset(client_train_input, client_train_target))
            client_test_datasets.append(SensorDataset(client_test_input, client_test_target))


    # the validation dataset contains the data from unseen subjects (the subjects never involve in training)
    # obj: check the global model performance on the unseen subjects
    unseen_client_val_datasets = [SensorDataset(
        val_inputs[val_subjects == sub], val_targets[val_subjects == sub]) 
        for sub in clients_val]


    # Create DataLoader objects
    client_train_loaders = [DataLoader(client_train_datasets[i], batch_size=BATCH_SIZE, shuffle=True) for i in range(len(clients_train))]
    client_test_loaders = [DataLoader(client_test_datasets[i], batch_size=BATCH_SIZE, shuffle=False) for i in range(len(clients_train))]
    unseen_client_val_loaders = [DataLoader(unseen_client_val_datasets[i], batch_size=1, shuffle=False) for i in range(len(clients_val))]

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


    # Create folder for results
    save_path = f"{SAVE_FOLDER}/{'FedProx' if args.fedprox else 'FedAvg'}/{args.binary_multi}/{datetime.now().strftime('%d-%m-%Y-%H%M%S')}"
    os.makedirs(save_path)

    overall_results = {}

    # Federated learning rounds
    for round in range(num_rounds):
        round_results = {}
        print(f"~ Federated Learning Round {round+1}/{num_rounds} ~")
        # Initialise list of individual local models
        local_models = []
        for i, client in enumerate(clients_train):
            # Create a new model instance and load the global model state dict
            local_model = LSTMModel(input_size, hidden_size, num_layers, num_classes).to(device)
            local_model.load_state_dict(global_model.state_dict())
            optimizer = optim.Adam(local_model.parameters(), lr=learning_rate)

            # Train the local model (normal or FedProx)
            # if args.fedprox:
            #     train_local_model_fedprox(global_model, local_model, client_dataloaders[client], criterion, optimizer, device, mu)
            # else:
            print(f"~ Training Local Model on client [{client}] ~")
            train_local_model(local_model, client_train_loaders[i], criterion, optimizer, device)

            local_models.append(local_model)
            
            results = evaluate_model(local_model, client_test_loaders[i], criterion, device, f"round_{round + 1}_local_{client}", save_path, args.binary_multi)
            round_results[f"local_client_{client}"] = results
            
            print(f"[{i}] Local Model {client}:\nAccuracy: {results['accuracy']}\nPrecision: {results['precision']}\nRecall: {results['recall']}\nF1 Score: {results['f1']}", end="\n\n")

        # Perform federated averaging to update the global model
        federated_averaging(global_model, local_models)
        
        # Evaluate global model on unseen clients
        global_results_accumulated = {}
        for i, unseen_client in enumerate(clients_val):
            print(f"~ Evaluating Global Model on Unseen Client [{unseen_client}] ~")
            results = evaluate_model(global_model, unseen_client_val_loaders[i], criterion, device, f"round_{round + 1}_global_unseen_{unseen_client}", save_path, args.binary_multi)
            round_results[f"global_unclient_{unseen_client}"] = results
            print(f"Global Model:\nAccuracy: {results['accuracy']}\nPrecision: {results['precision']}\nRecall: {results['recall']}\nF1 Score: {results['f1']}", end="\n\n")

            if i == 0:
                global_results_accumulated = results
            else:
                for key in results.keys():
                    global_results_accumulated[key] += results[key]

        # average the results of the global model on the unseen clients
        for key in global_results_accumulated.keys():
            global_results_accumulated[key] = global_results_accumulated[key]/len(clients_val) if key != 'class_report' else None
        round_results["global_unseen"] = global_results_accumulated

        # Evaluate global model on seen clients (but on the test set)
        global_results_accumulated = {}
        for i, client in enumerate(clients_train):
            print(f"~ Evaluating Global Model on Seen Client [{client}] ~")
            results = evaluate_model(global_model, client_test_loaders[i], criterion, device, f"round_{round + 1}_global_seen_{client}", save_path, args.binary_multi)
            round_results[f"global_client_{client}"] = results
            print(f"Global Model:\nAccuracy: {results['accuracy']}\nPrecision: {results['precision']}\nRecall: {results['recall']}\nF1 Score: {results['f1']}", end="\n\n")
        
            if i == 0:
                global_results_accumulated = results
            else:
                for key in results.keys():
                    global_results_accumulated[key] = global_results_accumulated[key] + results[key] if key != 'class_report' else None

        # average the results of the global model on the seen clients
        for key in global_results_accumulated.keys():
            global_results_accumulated[key] = global_results_accumulated[key]/len(clients_val) if key != 'class_report' else None
        round_results["global_seen"] = global_results_accumulated

        # average the results for both seen and unseen clients
        round_results["global"] = [(round_results["global_seen"][key] + round_results["global_unseen"][key])/2 
                                  if key != "class_report" else None
                                  for key in round_results["global_seen"].keys()]
        
        
        overall_results[f"round_{round + 1}"] = round_results

    # Export results
    with open(f"{save_path}/results.json", 'w') as json_file:
        json.dump(overall_results, json_file, indent=4)

        
# Program entrypoint
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Federated Learning with FedProx')
    parser.add_argument('--fedprox', action='store_true', help='Use FedProx algorithm')
    parser.add_argument('--binary_multi', default='binary', help='Binary or multi-class classification')
    args = parser.parse_args()
    main(args)
