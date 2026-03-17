import os
import torch
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import interp1d
from sklearn.metrics import r2_score
from model import KneePINN
from data_loader import create_dataloaders

def evaluate_model():
    data_dir = r"c:/Users/bennn/localSchool/PINN/PINN-MME4469/data/Overground Gait Trials"
    trial_names = ['DM_ngait_og1', 'DM_ngait_og2', 'DM_ngait_og3', 'DM_ngait_og4']
    
    # We load data with batch_size equal to the whole validation set for easy processing
    _, val_loader, scalers = create_dataloaders(data_dir, trial_names, batch_size=2048)
    
    device = torch.device('cpu')
    model_path = 'results/best_pinn_model.pth'
    
    input_dim = val_loader.dataset.feature_dim
    output_dim = val_loader.dataset.target_dim
    feature_names = val_loader.dataset.feature_names
    
    model = KneePINN(input_dim, output_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    all_preds, all_trues = [], []
    with torch.no_grad():
        for inputs, targets in val_loader:
            preds = model(inputs)
            # Flatten LSTM sequence outputs for standard array evaluation
            all_preds.append(preds.reshape(-1, preds.shape[-1]).numpy())
            all_trues.append(targets.reshape(-1, targets.shape[-1]).numpy())
            
    # Flatten
    preds_scaled = np.vstack(all_preds)
    trues_scaled = np.vstack(all_trues)
    
    # Unscale
    target_scaler = scalers['targets']
    preds_unscaled = target_scaler.inverse_transform(preds_scaled)
    trues_unscaled = target_scaler.inverse_transform(trues_scaled)
    
    # Target columns: ['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz']
    # 1. Correlation Analysis (R^2 and Scatter)
    os.makedirs('results/plots', exist_ok=True)
    
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(['Fx', 'Fy', 'Fz']):
        y_true = trues_unscaled[:, i]
        y_pred = preds_unscaled[:, i]
        r2 = r2_score(y_true, y_pred)
        
        # Scatter
        plt.subplot(2, 3, i+1)
        plt.scatter(y_true, y_pred, alpha=0.5, s=10)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
        plt.title(f'{col} (R² = {r2:.3f})')
        plt.xlabel('True Force (N/BW)')
        plt.ylabel('Predicted Force')
        
        # Residuals
        plt.subplot(2, 3, i+4)
        residuals = y_true - y_pred
        sns.histplot(residuals, kde=True)
        plt.title(f'{col} Residuals')
        plt.xlabel('Error')
        
    plt.tight_layout()
    plt.savefig('results/plots/correlation_residuals.png')
    
    # 2. Time-Series Plots (1-100% gait cycle representation)
    # We'll take the first 100 frames as a proxy for the gait cycle if it's long enough,
    # or interpolate the entire validation sequence to 100 points.
    seq_len = min(len(trues_unscaled), 500) # take a chunk
    y_true_chunk = trues_unscaled[:seq_len, 2] # Fz
    y_pred_chunk = preds_unscaled[:seq_len, 2] # Fz
    
    x_old = np.linspace(0, 100, seq_len)
    x_new = np.linspace(1, 100, 100)
    
    f_interp_true = interp1d(x_old, y_true_chunk, kind='cubic')
    f_interp_pred = interp1d(x_old, y_pred_chunk, kind='cubic')
    
    fz_true_100 = f_interp_true(x_new)
    fz_pred_100 = f_interp_pred(x_new)
    
    # Deriving Medial/Lateral algebraically from Fz and Tx as per our Physics Loss constraint
    condyle_distance = 0.04
    tx_pred_chunk = preds_unscaled[:seq_len, 3]
    f_interp_tx = interp1d(x_old, tx_pred_chunk, kind='cubic')
    tx_pred_100 = f_interp_tx(x_new)
    
    medial_pred_100 = (fz_pred_100 / 2.0) + (tx_pred_100 / condyle_distance)
    lateral_pred_100 = (fz_pred_100 / 2.0) - (tx_pred_100 / condyle_distance)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_new, fz_true_100, 'k--', label='True Total Axial Force (Fz)', linewidth=2)
    plt.plot(x_new, fz_pred_100, 'b-', label='Predicted Total Axial Force (Fz)')
    plt.plot(x_new, medial_pred_100, 'g-', label='Predicted Medial Force', alpha=0.8)
    plt.plot(x_new, lateral_pred_100, 'r-', label='Predicted Lateral Force', alpha=0.8)
    plt.title('Knee Contact Forces over Gait Cycle')
    plt.xlabel('% Gait Cycle')
    plt.ylabel('Force')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/plots/gait_cycle_forces.png')

    # 3. Export to SimTK Excel Format
    df_export = pd.DataFrame({
        '% Gait Cycle': x_new,
        'Predicted_Fz': fz_pred_100,
        'Predicted_F_Medial': medial_pred_100,
        'Predicted_F_Lateral': lateral_pred_100
    })
    df_export.to_excel('results/SimTK_Predictions.xlsx', index=False)
    
    # 4. Feature Importance (Weight magnitude of LSTM input-hidden weights)
    # For LSTM, weight_ih_l0 has shape (4*hidden_size, input_size) — we average across all gates
    first_layer_weights = model.lstm.weight_ih_l0.data.abs().mean(dim=0).numpy()
    
    # Group features by type
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
    plt.title('Feature Group Importance (Absolute First-Layer Weights)')
    plt.ylabel('Aggregated Weight Magnitude')
    plt.savefig('results/plots/feature_importance.png')
    
    print("Evaluation Complete. Plots and Excel File saved to results/")

if __name__ == "__main__":
    evaluate_model()
