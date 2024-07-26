import networks 
from archived.utilsv1d.utilsv1 import *
import torch 
import torch.nn as nn 
import torch.optim as optim
import argparse, os, json
import warnings
warnings.filterwarnings("ignore")


def main(args):
    if args.model == 'LSTMModel':
        model = networks.LSTMModel
    elif args.model == 'CNNLSTM':
        model = networks.CNNLSTM 
    elif args.model == "LSTMWeighted":
        model = networks.LSTMWeighted
    elif args.model == "LSTMWeightedModalityFusion":
        model = networks.LSTMWeightedModalityFusion
    elif args.model =="LSTMWeightedModalityFeatureFusion":
        model = networks.LSTMWeightedModalityFeatureFusion

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

    if args.binary_multi == "multi" or args.binary_multi == "both":
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
    num_classes = len(scenario_mapping)  #len(np.unique(train_targets))
    hidden_size = 128
    num_layers = 2
    learning_rate = 0.01
    num_epochs = 10 # Reduced local epochs
    num_clients = 10
    num_rounds = 10
    mu = 0.01  # FedProx hyperparameter

    # Initialize global model, criterion, and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    global_model = {
        "binary_lstm": model(input_size, hidden_size, num_layers, 2).to(device),
        "fall_lstm": model(input_size, hidden_size, num_layers, num_classes).to(device),
        "non_fall_lstm": model(input_size, hidden_size, num_layers, num_classes).to(device)
    }
    criterion = nn.CrossEntropyLoss()
    

    # Create folder for results
    save_path = f"{SAVE_FOLDER}/{'FedProx' if args.fedprox else 'FedAvg'}/{args.binary_multi}_hierarchical_{args.model}/{datetime.now().strftime('%d-%m-%Y-%H%M%S')}"
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
            local_model = {
                "binary_lstm": model(input_size, hidden_size, num_layers, 2).to(device),
                "fall_lstm": model(input_size, hidden_size, num_layers, num_classes).to(device),
                "non_fall_lstm": model(input_size, hidden_size, num_layers, num_classes).to(device)
            }
            optimizer = {}
            for k in local_model.keys():
                local_model[k].load_state_dict(global_model[k].state_dict())
                optimizer[k] = optim.Adam(local_model[k].parameters(), lr=learning_rate)

            # Train the local model (normal or FedProx)
            print(f"~ Training Local Model on client [{client}] ~")
            if args.fedprox:
                local_model = train_local_model_fedprox_hierarchical(global_model, local_model, client_train_loaders[i], criterion, optimizer, device, mu)
            else:
                local_model = train_local_model_hierarchical(local_model, client_train_loaders[i], criterion, optimizer, device)

            local_models.append(local_model)
            
            results_binary, results_multi = evaluate_hierarchical_model(local_model, client_test_loaders[i], criterion, device, f"round_{round + 1}_local_{client}", save_path)
            round_results[f"local_client_{client}_binary"] = results_binary
            round_results[f"local_client_{client}_multi"] = results_multi
            
            print(f"[{i}] Local Model {client} - Binary:\nAccuracy: {results_binary['accuracy']}\nPrecision: {results_binary['precision']}\nRecall: {results_binary['recall']}\nF1 Score: {results_binary['f1']}", end="\n\n")
            print(f"[{i}] Local Model {client} - Multi:\nAccuracy: {results_multi['accuracy']}\nPrecision: {results_multi['precision']}\nRecall: {results_multi['recall']}\nF1 Score: {results_multi['f1']}", end="\n\n")
            # print(f"[{i}] Local Model {client}:\nAccuracy: {results['accuracy']}\nPrecision: {results['precision']}\nRecall: {results['recall']}\nF1 Score: {results['f1']}", end="\n\n")

        # Perform federated averaging to update the global model
        federated_averaging_hierarchical(global_model, local_models)
        
        # Evaluate global model on unseen clients
        global_results_accumulated_binary = {}
        global_results_accumulated_multi = {}
        for i, unseen_client in enumerate(clients_val):
            print(f"~ Evaluating Global Model on Unseen Client [{unseen_client}] ~")
            results_binary, results_multi = evaluate_hierarchical_model(global_model, unseen_client_val_loaders[i], criterion, device, f"round_{round + 1}_global_unseen_{unseen_client}", save_path)
            round_results[f"global_unclient_{unseen_client}_binary"] = results_binary
            round_results[f"global_unclient_{unseen_client}_multi"] = results_multi
            print(f"Global Model - Binary:\nAccuracy: {results_binary['accuracy']}\nPrecision: {results_binary['precision']}\nRecall: {results_binary['recall']}\nF1 Score: {results_binary['f1']}", end="\n\n")
            print(f"Global Model - Multi:\nAccuracy: {results_multi['accuracy']}\nPrecision: {results_multi['precision']}\nRecall: {results_multi['recall']}\nF1 Score: {results_multi['f1']}", end="\n\n")
            # print(f"Global Model:\nAccuracy: {results['accuracy']}\nPrecision: {results['precision']}\nRecall: {results['recall']}\nF1 Score: {results['f1']}", end="\n\n")

            if i == 0:
                global_results_accumulated_binary = results_binary
                global_results_accumulated_multi = results_multi
            else:
                for key in results_binary.keys():
                    global_results_accumulated_binary[key] += results_binary[key]
                    global_results_accumulated_multi[key] += results_multi[key]

        # average the results of the global model on the unseen clients
        for key in global_results_accumulated_binary.keys():
            global_results_accumulated_binary[key] = global_results_accumulated_binary[key]/len(clients_val) if key != 'class_report' else None
            global_results_accumulated_multi[key] = global_results_accumulated_multi[key]/len(clients_val) if key != 'class_report' else None
        round_results["global_unseen_binary"] = global_results_accumulated_binary
        round_results["global_unseen_multi"] = global_results_accumulated_multi

        # Evaluate global model on seen clients (but on the test set)
        global_results_accumulated_binary = {}
        global_results_accumulated_multi = {}
        for i, client in enumerate(clients_train):
            print(f"~ Evaluating Global Model on Seen Client [{client}] ~")
            results_binary, results_multi = evaluate_hierarchical_model(global_model, client_test_loaders[i], criterion, device, f"round_{round + 1}_global_seen_{client}", save_path)
            round_results[f"global_client_{client}_binary"] = results_binary
            round_results[f"global_client_{client}_multi"] = results_multi
            print(f"Global Model - Binary:\nAccuracy: {results_binary['accuracy']}\nPrecision: {results_binary['precision']}\nRecall: {results_binary['recall']}\nF1 Score: {results_binary['f1']}", end="\n\n")
            print(f"Global Model - Multi:\nAccuracy: {results_multi['accuracy']}\nPrecision: {results_multi['precision']}\nRecall: {results_multi['recall']}\nF1 Score: {results_multi['f1']}", end="\n\n")
            # print(f"Global Model:\nAccuracy: {results['accuracy']}\nPrecision: {results['precision']}\nRecall: {results['recall']}\nF1 Score: {results['f1']}", end="\n\n")
        
            if i == 0:
                global_results_accumulated_binary = results_binary
                global_results_accumulated_multi = results_multi
            else:
                for key in results_binary.keys():
                    global_results_accumulated_binary[key] = global_results_accumulated_binary[key] + results_binary[key] if key != 'class_report' else None
                    global_results_accumulated_multi[key] = global_results_accumulated_multi[key] + results_multi[key] if key != 'class_report' else None

        # average the results of the global model on the seen clients
        for key in global_results_accumulated_binary.keys():
            global_results_accumulated_binary[key] = global_results_accumulated_binary[key]/len(clients_val) if key != 'class_report' else None
            global_results_accumulated_multi[key] = global_results_accumulated_multi[key]/len(clients_val) if key != 'class_report' else None
        round_results["global_seen_binary"] = global_results_accumulated_binary
        round_results["global_seen_multi"] = global_results_accumulated_multi

        # average the results for both seen and unseen clients
        round_results["global_binary"] = [(round_results["global_seen_binary"][key] + round_results["global_unseen_binary"][key])/2 
                                  if key != "class_report" else None
                                  for key in round_results["global_seen_binary"].keys()]
        round_results["global_multi"] = [(round_results["global_seen_multi"][key] + round_results["global_unseen_multi"][key])/2 
                                  if key != "class_report" else None
                                  for key in round_results["global_seen_multi"].keys()]
        
        
        overall_results[f"round_{round + 1}"] = round_results

    # Export results
    with open(f"{save_path}/results.json", 'w') as json_file:
        json.dump(overall_results, json_file, indent=4)

        
# Program entrypoint
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Federated Learning with FedProx')
    parser.add_argument('--fedprox', action='store_true', help='Use FedProx algorithm')
    parser.add_argument('--binary_multi', default='binary', help='Binary or multi-class classification')
    parser.add_argument('--model', default='binary', help='LSTMModel, CNNLSTM, LSTMWeighted, LSTMWeightedModalityFusion, LSTMWeightedModalityFeatureFusion')
    args = parser.parse_args()

    main(args)
