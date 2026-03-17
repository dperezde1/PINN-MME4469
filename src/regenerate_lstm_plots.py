"""
One-off script to regenerate LSTM evaluation plots into results/lstm_plots/
Uses the archived LSTM model + data loader from src/archive/lstm/
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'archive', 'lstm'))

import torch
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import interp1d
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Import LSTM-specific modules from archive
from model_lstm import KneePINN
from data_loader_lstm import create_dataloaders

def regenerate_lstm_plots():
    data_dir = r"c:/Users/bennn/localSchool/PINN/PINN-MME4469/data/Overground Gait Trials"
    trial_names = ['DM_ngait_og1', 'DM_ngait_og2', 'DM_ngait_og3', 'DM_ngait_og4']
    
    _, val_loader, scalers = create_dataloaders(data_dir, trial_names, batch_size=2048)
    
    device = torch.device('cpu')
    model_path = 'src/archive/lstm/best_lstm_model.pth'
    
    input_dim = val_loader.dataset.feature_dim
    output_dim = val_loader.dataset.target_dim
    
    model = KneePINN(input_dim=input_dim, output_dim=output_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    all_preds, all_trues = [], []
    with torch.no_grad():
        for inputs, targets in val_loader:
            preds = model(inputs)
            all_preds.append(preds.reshape(-1, preds.shape[-1]).numpy())
            all_trues.append(targets.reshape(-1, targets.shape[-1]).numpy())
            
    preds_scaled = np.vstack(all_preds)
    trues_scaled = np.vstack(all_trues)
    
    preds_unscaled = scalers['targets'].inverse_transform(preds_scaled)
    trues_unscaled = scalers['targets'].inverse_transform(trues_scaled)
    
    out_dir = 'results/lstm_plots'
    os.makedirs(out_dir, exist_ok=True)
    
    # Correlation + Residuals
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(['Fx', 'Fy', 'Fz']):
        r2 = r2_score(trues_unscaled[:, i], preds_unscaled[:, i])
        plt.subplot(2, 3, i+1)
        plt.scatter(trues_unscaled[:, i], preds_unscaled[:, i], alpha=0.4, s=10)
        plt.plot([trues_unscaled[:, i].min(), trues_unscaled[:, i].max()],
                 [trues_unscaled[:, i].min(), trues_unscaled[:, i].max()], 'r--')
        plt.title(f'{col} (R² = {r2:.3f})')
        plt.xlabel('True Force (N/BW)')
        plt.ylabel('Predicted Force')
        
        plt.subplot(2, 3, i+4)
        residuals = preds_unscaled[:, i] - trues_unscaled[:, i]
        sns.histplot(residuals, kde=True)
        plt.title(f'{col} Residuals')
        plt.xlabel('Error')
        
    plt.tight_layout()
    plt.savefig(f'{out_dir}/correlation_residuals.png')
    plt.close()
    
    # Gait cycle forces
    n = min(100, len(preds_unscaled))
    gait_pct = np.linspace(1, 100, n)
    
    fz_pred = preds_unscaled[:n, 2]
    fz_true = trues_unscaled[:n, 2]
    tx_pred = preds_unscaled[:n, 3]
    d = 0.04
    medial_pred = (fz_pred / 2.0) + (tx_pred / d)
    lateral_pred = (fz_pred / 2.0) - (tx_pred / d)
    
    plt.figure(figsize=(10, 6))
    plt.plot(gait_pct, fz_true, 'k--', lw=2, label='True Total Axial Force (Fz)')
    plt.plot(gait_pct, fz_pred, 'b-', lw=2, label='Predicted Total Axial Force (Fz)')
    plt.plot(gait_pct, medial_pred, 'g-', lw=1.5, label='Predicted Medial Force')
    plt.plot(gait_pct, lateral_pred, 'r-', lw=1.5, label='Predicted Lateral Force')
    plt.title('LSTM: Knee Contact Forces over Gait Cycle')
    plt.xlabel('% Gait Cycle')
    plt.ylabel('Force')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{out_dir}/gait_cycle_forces.png')
    plt.close()
    
    # Feature importance  
    first_layer_weights = model.lstm.weight_ih_l0.data.abs().mean(dim=0).numpy()
    feature_names = val_loader.dataset.feature_names
    
    emg_idx = [i for i, f in enumerate(feature_names) if 'emg' in f.lower() or f in ['semimem', 'bifem', 'recfem', 'vasmed', 'vaslat', 'medgas', 'latgas', 'soleus', 'tibant', 'perlng', 'gmax', 'gmed', 'sartorius']]
    grf_idx = [i for i, f in enumerate(feature_names) if 'grf' in f.lower() or 'cop' in f.lower() or len(f) <= 3]
    kin_idx = [i for i in range(len(feature_names)) if i not in emg_idx and i not in grf_idx]
    
    importance = {
        'EMG': np.sum(first_layer_weights[emg_idx]) if emg_idx else 0,
        'GRF': np.sum(first_layer_weights[grf_idx]) if grf_idx else 0,
        'Kinematics': np.sum(first_layer_weights[kin_idx]) if kin_idx else 0
    }
    
    plt.figure(figsize=(8, 5))
    plt.bar(importance.keys(), importance.values(), color=['orange', 'green', 'purple'])
    plt.title('LSTM: Feature Group Importance')
    plt.ylabel('Aggregated Weight Magnitude')
    plt.savefig(f'{out_dir}/feature_importance.png')
    plt.close()
    
    print(f"LSTM plots regenerated in {out_dir}/")
    
    # Print metrics for comparison
    for i, col in enumerate(['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz']):
        rmse = np.sqrt(mean_squared_error(trues_unscaled[:, i], preds_unscaled[:, i]))
        mae = mean_absolute_error(trues_unscaled[:, i], preds_unscaled[:, i])
        r2 = r2_score(trues_unscaled[:, i], preds_unscaled[:, i])
        print(f"  {col}: RMSE={rmse:.2f}, MAE={mae:.2f}, R2={r2:.3f}")

if __name__ == "__main__":
    regenerate_lstm_plots()
