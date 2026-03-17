import torch
import torch.nn as nn

class KneePINN(nn.Module):
    """
    Patient-Specific Physics-Informed Neural Network for Knee JRF Prediction.
    Deliberately compact architecture ([32, 16]) to prevent overfitting on small datasets.
    """
    def __init__(self, input_dim, output_dim=6, hidden_layers=[32, 16]):
        super(KneePINN, self).__init__()
        
        layers = []
        in_features = input_dim
        
        for h in hidden_layers:
            layers.append(nn.Linear(in_features, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ELU())
            in_features = h
            
        self.feature_extractor = nn.Sequential(*layers)
        self.output_layer = nn.Linear(in_features, output_dim)
        
    def forward(self, x):
        features = self.feature_extractor(x)
        preds = self.output_layer(features)
        return preds
