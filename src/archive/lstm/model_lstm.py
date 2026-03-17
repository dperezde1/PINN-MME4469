import torch
import torch.nn as nn

class KneePINN(nn.Module):
    """
    Physics-Informed Neural Network to predict Knee Joint Reaction Forces (JRF).
    Outputs 6 components (Fx, Fy, Fz, Tx, Ty, Tz) to match the eTibia target data.
    Upgraded to an LSTM architecture for temporal sequence modeling.
    """
    def __init__(self, input_dim, output_dim=6, lstm_hidden=128, lstm_layers=2, hidden_layers=[128, 64]):
        super(KneePINN, self).__init__()
        
        # LSTM layer to capture temporal dependencies
        self.lstm = nn.LSTM(
            input_size=input_dim, 
            hidden_size=lstm_hidden, 
            num_layers=lstm_layers, 
            batch_first=True,
            dropout=0.2 if lstm_layers > 1 else 0
        )
        
        # MLP Feature Extractor for the LSTM outputs at each time step
        layers = []
        in_features = lstm_hidden
        
        for h in hidden_layers:
            layers.append(nn.Linear(in_features, h))
            layers.append(nn.ELU()) 
            layers.append(nn.Dropout(0.1))
            in_features = h
            
        self.feature_extractor = nn.Sequential(*layers)
        
        # Output layer for the 6 target variables
        self.output_layer = nn.Linear(in_features, output_dim)
        
    def forward(self, x):
        # x shape: (batch_size, seq_length, input_dim)
        # If input is 2D (batch, features), unsqueeze it
        if x.dim() == 2:
            x = x.unsqueeze(1)
            
        lstm_out, _ = self.lstm(x)
        # lstm_out shape: (batch_size, seq_length, lstm_hidden)
        
        # Apply MLP to every time step seamlessly using PyTorch's implicit broadcasting
        features = self.feature_extractor(lstm_out)
        preds = self.output_layer(features)
        
        # Squeeze out sequence dimension if it was originally 2D
        if preds.shape[1] == 1:
            preds = preds.squeeze(1)
            
        return preds
