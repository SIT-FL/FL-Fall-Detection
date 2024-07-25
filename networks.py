import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
    
# Simple LSTM model for experimental purposes
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, h0 = None, c0 = None):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device) if h0 is None else h0
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device) if c0 is None else c0
        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out, (hn, cn)

# cascaded  approach 
class CLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(CLSTM, self).__init__()
        self.base_lstm = LSTMModel(input_size, hidden_size, num_layers, 2)
        self.last_lstm = LSTMModel(input_size+2, hidden_size, num_layers, num_classes)


    def forward(self, x):
        binary_out, (hn, cn) = self.base_lstm(x)

        # concate the binary_out to the input
        x = torch.cat((x, binary_out.unsqueeze(1).repeat(1, x.size(1), 1)), dim=2)
        out, _ = self.last_lstm(x, hn, cn)
                                  
        return binary_out, out

class CNNLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, input_channels=11):
        super(CNNLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Define the 1D CNN layers
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.conv4 = nn.Conv1d(64, input_channels, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(input_channels)

        # Define the LSTM layers
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.lstm3 = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)

        # Define the fully connected layer
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, hn=None, cn=None):
        x = x.permute(0, 2, 1)  # (batch_size, input_channels, sequence_length)
        # Apply 1D CNN layers with batch normalization, ReLU, and dropout
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.dropout(x, p=0.5, training=self.training)

        # Reshape for LSTM layers
        x = x.permute(0, 2, 1)  # (batch_size, sequence_length, input_size)
        # Initialize hidden and cell states if not provided
        if hn is None or cn is None:
            hn = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            cn = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Apply LSTM layers with dropout
        x, (hn, cn) = self.lstm1(x, (hn, cn))
        x = F.dropout(x, p=0.5, training=self.training)
        x, (hn, cn) = self.lstm2(x, (hn, cn))
        x = F.dropout(x, p=0.5, training=self.training)
        x, (hn, cn) = self.lstm3(x, (hn, cn))
        x = F.dropout(x, p=0.5, training=self.training)

        # Flatten and apply fully connected layer
        out = self.fc(x[:, -1, :])

        return out, (hn, cn)

class LSTMWeighted(nn.Module):
    def __init__(self, num_features, hidden_size, num_layers, num_classes, desired_seq_length=512):
        super(LSTMWeighted, self).__init__()
        self.desired_seq_length = desired_seq_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(num_features, hidden_size, num_layers, batch_first=True)

        # weighted
        self.weighted = nn.Linear(self.desired_seq_length, 1)

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        b, seq_length, num_features = x.size()  
        # downsample/updsample the sequence length to 512   
        x = x.permute(0, 2, 1)
        x = F.interpolate(x, size=self.desired_seq_length, mode='linear')
        x = x.permute(0, 2, 1)

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, (hn, cn) = self.lstm(x, (h0, c0))

        # instead of taking the output embedding at the last time step, we can apply a weighted average to all the sequence, where the weights are learned
        out = self.weighted(out.permute(0, 2, 1)).squeeze(2)

        #flatten the output
        out = out.view(out.size(0), -1)

        out = self.fc(out)
        return out, (hn, cn)

class LSTMWeightedModalityFusion(nn.Module):
    def __init__(self, num_features, hidden_size, num_layers, num_classes, desired_seq_length=512):
        super(LSTMWeightedModalityFusion, self).__init__()
        self.desired_seq_length = desired_seq_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.modal1 = LSTMWeighted(4, hidden_size, num_layers, num_classes, desired_seq_length)  # LSTM for the first modality, w,x,y,z
        self.modal2 = LSTMWeighted(3, hidden_size, num_layers, num_classes, desired_seq_length)  # LSTM for the second modality, droll, dpitch, dyaw
        self.modal3 = LSTMWeighted(3, hidden_size, num_layers, num_classes, desired_seq_length)  # LSTM for the third modality, acc_x,acc_y,acc_z

        self.fc = nn.Linear(num_classes*3, num_classes)

    def forward(self, x):
        b, seq_length, num_features = x.size()  
        # downsample/updsample the sequence length to 512   
        x = x.permute(0, 2, 1)
        x = F.interpolate(x, size=self.desired_seq_length, mode='linear')
        x = x.permute(0, 2, 1)

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out1, _ = self.modal1(x[:, :, :4])
        out2, _ = self.modal2(x[:, :, 4:7])
        out3, _ = self.modal3(x[:, :, 7:-1])

        # consider the output from the three modalities, and fuse them with fc
        out = self.fc(torch.cat((out1, out2, out3), dim=1))
        return out, (_, _)

class LSTMWeightedFeature(nn.Module):
    def __init__(self, num_features, hidden_size, num_layers, num_classes, desired_seq_length=512):
        super(LSTMWeightedFeature, self).__init__()
        self.desired_seq_length = desired_seq_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(num_features, hidden_size, num_layers, batch_first=True)

        # weighted
        self.weighted = nn.Linear(self.desired_seq_length, 1)

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        b, seq_length, num_features = x.size()  
        # downsample/updsample the sequence length to 512   
        x = x.permute(0, 2, 1)
        x = F.interpolate(x, size=self.desired_seq_length, mode='linear')
        x = x.permute(0, 2, 1)

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        # print(f"x: {x.shape}")
        out, _ = self.lstm(x, (h0, c0))

        # instead of taking the output embedding at the last time step, we can apply a weighted average to all the sequence, where the weights are learned
        out = self.weighted(out.permute(0, 2, 1)).squeeze(2)

        #flatten the output
        out = out.view(out.size(0), -1)
        return out

class LSTMWeightedModalityFeatureFusion(nn.Module):
    def __init__(self, num_features, hidden_size, num_layers, num_classes, desired_seq_length=512):
        super(LSTMWeightedModalityFeatureFusion, self).__init__()
        self.desired_seq_length = desired_seq_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.modal1 = LSTMWeightedFeature(4, hidden_size, num_layers, num_classes, desired_seq_length)  # LSTM for the first modality, w,x,y,z
        self.modal2 = LSTMWeightedFeature(3, hidden_size, num_layers, num_classes, desired_seq_length)  # LSTM for the second modality, droll, dpitch, dyaw
        self.modal3 = LSTMWeightedFeature(3, hidden_size, num_layers, num_classes, desired_seq_length)  # LSTM for the third modality, acc_x,acc_y,acc_z

        # remove the last layer 

        self.fc1 = nn.Linear(hidden_size*3, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        b, seq_length, num_features = x.size()  
        # downsample/updsample the sequence length to 512   
        x = x.permute(0, 2, 1)
        x = F.interpolate(x, size=self.desired_seq_length, mode='linear')
        x = x.permute(0, 2, 1)

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out1 = self.modal1(x[:, :, :4])
        out2 = self.modal2(x[:, :, 4:7])
        out3 = self.modal3(x[:, :, 7:-1])

        # consider the output from the three modalities, and fuse them with fc
        out = self.fc1(torch.cat((out1, out2, out3), dim=1))
        out = self.fc2(out)
        
        return out, (h0, c0)
