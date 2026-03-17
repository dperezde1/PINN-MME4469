import torch
import torch.nn as nn

class KneePINN(nn.Module):
    """
    Physics-Informed Neural Network to predict Knee Joint Reaction Forces (JRF).
    Outputs 6 components (Fx, Fy, Fz, Tx, Ty, Tz) to match the eTibia target data.
    """
    def __init__(self, input_dim, output_dim=6, hidden_layers=[256, 256, 128, 64]):
        super(KneePINN, self).__init__()
        
        layers = []
        in_features = input_dim
        
        for h in hidden_layers:
            layers.append(nn.Linear(in_features, h))
            layers.append(nn.ELU()) # ELU provides smoother gradients for physics loss
            # Optional: Add Dropout or BatchNorm if overfitting
            # layers.append(nn.Dropout(0.1))
            in_features = h
            
        self.feature_extractor = nn.Sequential(*layers)
        
        # Output layer for the 6 target variables
        self.output_layer = nn.Linear(in_features, output_dim)
        
    def forward(self, x):
        features = self.feature_extractor(x)
        preds = self.output_layer(features)
        return preds
