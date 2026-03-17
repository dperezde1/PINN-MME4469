"""
Phase 7: Patient-Specific Overfit Model
Trains on ALL 4 trials with no holdout set.
Justified: This produces a patient-specific calibration model for per-patient deployment.
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

sys.path.insert(0, os.path.dirname(__file__))
from data_loader import SimTKDataset, PATIENT_BW
from model import KneePINN
from physics_loss import PhysicalConstraintsLoss

# Colors
C_BG = '#0D1117'; C_CARD = '#161B22'; C_GRID = '#21262D'
C_TEXT = '#C9D1D9'; C_SUB = '#8B949E'
C_BLUE = '#58A6FF'; C_GREEN = '#3FB950'; C_RED = '#F85149'
C_PURPLE = '#D2A8FF'; C_ORANGE = '#F0883E'

DATA_DIR = r"c:/Users/bennn/localSchool/PINN/PINN-MME4469/data/Overground Gait Trials"
TRIALS = ['DM_ngait_og1', 'DM_ngait_og2', 'DM_ngait_og3', 'DM_ngait_og4']
OUT = 'results/patient_specific'
os.makedirs(OUT, exist_ok=True)
os.makedirs(f'{OUT}/plots', exist_ok=True)


def train_overfit():
    print("=" * 60)
    print("Phase 7: Patient-Specific Overfit Training")
    print("Training on ALL 4 trials (no validation holdout)")
    print("=" * 60)
    
    # Use ALL trials for training with more PCA components (since we want to overfit)
    dataset = SimTKDataset(DATA_DIR, TRIALS, is_train=True, n_components=25, 
                            scaler_save_dir=OUT)
    loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True, drop_last=False)
    
    print(f"  Total samples: {len(dataset)}")
    print(f"  Features: {dataset.feature_dim}, Targets: {dataset.target_dim}")
    
    # Larger model for intentional overfitting - no BatchNorm to allow more memorization
    model = KneePINN(input_dim=dataset.feature_dim, output_dim=dataset.target_dim,
                     hidden_layers=[64, 64, 32])
    
    # Lower physics weight to prioritize data fidelity
    optimizer = optim.Adam(model.parameters(), lr=5e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500, eta_min=1e-5)
    criterion = PhysicalConstraintsLoss(lambda_physics=0.1)
    
    history = []
    best_loss = float('inf')
    
    print(f"\n  Training for 500 epochs (no early stopping)...")
    for epoch in range(500):
        model.train()
        epoch_loss = 0
        for x, y in loader:
            optimizer.zero_grad()
            loss, dl, pl = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(loader)
        history.append(avg_loss)
        scheduler.step()
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), f'{OUT}/patient_model.pth')
        
        if epoch % 25 == 0 or epoch == 499:
            print(f"    Epoch [{epoch}/500] Loss: {avg_loss:.4f}")
    
    print(f"\n  Training complete. Best loss: {best_loss:.4f}")
    
    # Learning curve
    fig, ax = plt.subplots(figsize=(10, 4), facecolor=C_BG)
    ax.set_facecolor(C_CARD)
    ax.plot(history, color=C_BLUE, lw=1.5)
    ax.set_xlabel('Epoch', color=C_TEXT)
    ax.set_ylabel('Loss', color=C_TEXT)
    ax.set_title('Patient-Specific Model: Training Loss', color=C_TEXT, fontsize=12, fontweight='bold')
    ax.tick_params(colors=C_SUB)
    ax.spines['bottom'].set_color(C_GRID); ax.spines['left'].set_color(C_GRID)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.grid(color=C_GRID, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUT}/plots/learning_curve.png', dpi=150, facecolor=C_BG, bbox_inches='tight')
    plt.close()
    
    return model, dataset


def evaluate_overfit(model, dataset):
    print("\n" + "=" * 60)
    print("Evaluating (on training data - expected to fit well)")
    print("=" * 60)
    
    model.load_state_dict(torch.load(f'{OUT}/patient_model.pth', map_location='cpu'))
    model.eval()
    
    loader = torch.utils.data.DataLoader(dataset, batch_size=2048, shuffle=False)
    all_p, all_t = [], []
    with torch.no_grad():
        for x, y in loader:
            all_p.append(model(x).numpy())
            all_t.append(y.numpy())
    
    preds = dataset.scalers['targets'].inverse_transform(np.vstack(all_p))
    trues = dataset.scalers['targets'].inverse_transform(np.vstack(all_t))
    
    cols = ['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz']
    print(f"\n  {'Component':>10} {'RMSE':>10} {'MAE':>10} {'R2':>10}")
    print("  " + "-" * 45)
    metrics = {}
    for i, col in enumerate(cols):
        rmse = np.sqrt(mean_squared_error(trues[:, i], preds[:, i]))
        mae = mean_absolute_error(trues[:, i], preds[:, i])
        r2 = r2_score(trues[:, i], preds[:, i])
        print(f"  {col:>10} {rmse:>10.2f} {mae:>10.2f} {r2:>10.3f}")
        metrics[col] = {'RMSE': rmse, 'MAE': mae, 'R2': r2}
    
    # Save metrics
    pd.DataFrame(metrics).T.to_csv(f'{OUT}/metrics.csv')
    
    # ── 1. Correlation + Residuals ──
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), facecolor=C_BG)
    for i, col in enumerate(['Fx', 'Fy', 'Fz']):
        r2 = metrics[col]['R2']
        ax = axes[0, i]; ax.set_facecolor(C_CARD)
        ax.scatter(trues[:, i], preds[:, i], alpha=0.5, s=12, color=C_BLUE)
        lims = [min(trues[:, i].min(), preds[:, i].min()),
                max(trues[:, i].max(), preds[:, i].max())]
        ax.plot(lims, lims, 'r--', lw=1.5)
        ax.set_title(f'{col} (R2 = {r2:.3f})', color=C_TEXT, fontsize=11, fontweight='bold')
        ax.set_xlabel('True Force (N)', color=C_SUB)
        ax.set_ylabel('Predicted Force (N)', color=C_SUB)
        ax.tick_params(colors=C_SUB)
        ax.spines['bottom'].set_color(C_GRID); ax.spines['left'].set_color(C_GRID)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.grid(color=C_GRID, alpha=0.3)
        
        ax2 = axes[1, i]; ax2.set_facecolor(C_CARD)
        residuals = preds[:, i] - trues[:, i]
        ax2.hist(residuals, bins=30, color=C_GREEN, alpha=0.7, edgecolor=C_GRID)
        ax2.axvline(x=0, color=C_RED, linestyle='--', lw=1.5)
        ax2.set_title(f'{col} Residuals', color=C_TEXT, fontsize=11, fontweight='bold')
        ax2.set_xlabel('Error (N)', color=C_SUB)
        ax2.tick_params(colors=C_SUB)
        ax2.spines['bottom'].set_color(C_GRID); ax2.spines['left'].set_color(C_GRID)
        ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f'{OUT}/plots/correlation_residuals.png', dpi=150, facecolor=C_BG, bbox_inches='tight')
    plt.close()
    
    # ── 2. BW-Normalized Gait Cycle ──
    n = min(100, len(preds))
    gait_pct = np.linspace(0, 100, n)
    fz_pred_bw = preds[:n, 2] / PATIENT_BW
    fz_true_bw = trues[:n, 2] / PATIENT_BW
    d = 0.04
    med_bw = ((preds[:n, 2] / 2.0) + (preds[:n, 3] / d)) / PATIENT_BW
    lat_bw = ((preds[:n, 2] / 2.0) - (preds[:n, 3] / d)) / PATIENT_BW
    
    phases = [('IC', 0, 2, C_ORANGE), ('LR', 2, 12, C_RED),
              ('MSt', 12, 31, C_GREEN), ('TSt', 31, 50, C_BLUE),
              ('PSw', 50, 62, C_PURPLE), ('Sw', 62, 100, '#FFA657')]
    
    fig, ax = plt.subplots(figsize=(12, 6), facecolor=C_BG)
    ax.set_facecolor(C_CARD)
    for name, s, e, color in phases:
        ax.axvspan(s, e, alpha=0.08, color=color)
        ax.text((s+e)/2, -0.3, name, ha='center', va='top', fontsize=8, color=color, fontweight='bold', alpha=0.7)
    
    ax.plot(gait_pct, fz_true_bw, 'w--', lw=2.5, label='True Fz (BW)', alpha=0.9)
    ax.plot(gait_pct, fz_pred_bw, color=C_BLUE, lw=2.5, label='Predicted Fz (BW)')
    ax.plot(gait_pct, med_bw, color=C_GREEN, lw=1.8, label='Predicted Medial (BW)', alpha=0.8)
    ax.plot(gait_pct, lat_bw, color=C_RED, lw=1.8, label='Predicted Lateral (BW)', alpha=0.8)
    
    ax.set_xlabel('% Gait Cycle', color=C_TEXT, fontsize=11)
    ax.set_ylabel('Force (x Body Weight)', color=C_TEXT, fontsize=11)
    ax.set_title('Patient-Specific Model: Knee Forces Normalized to BW (686.7 N)',
                color=C_TEXT, fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9, framealpha=0.3)
    ax.tick_params(colors=C_SUB)
    ax.spines['bottom'].set_color(C_GRID); ax.spines['left'].set_color(C_GRID)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.grid(color=C_GRID, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUT}/plots/bw_gait_cycle.png', dpi=150, facecolor=C_BG, bbox_inches='tight')
    plt.close()
    
    # ── 3. Physics Adherence (fixed scatter plot) ──
    fz = preds[:, 2]; tx = preds[:, 3]
    f_med = (fz / 2.0) + (tx / d)
    f_lat = (fz / 2.0) - (tx / d)
    
    med_violations = np.sum(f_med < 0)
    lat_violations = np.sum(f_lat < 0)
    total = len(f_med)
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor=C_BG)
    
    # Medial distribution
    ax = axes[0]; ax.set_facecolor(C_CARD)
    ax.hist(f_med / PATIENT_BW, bins=30, color=C_GREEN, alpha=0.7, edgecolor=C_GRID)
    ax.axvline(x=0, color=C_RED, linestyle='--', lw=2, label='Zero boundary')
    pct_ok = (total - med_violations) / total * 100
    ax.set_title(f'Medial Force (BW)\n{pct_ok:.0f}% compliant', color=C_TEXT, fontsize=11, fontweight='bold')
    ax.set_xlabel('Force (x BW)', color=C_TEXT); ax.set_ylabel('Count', color=C_TEXT)
    ax.legend(fontsize=8, framealpha=0.3)
    ax.tick_params(colors=C_SUB)
    ax.spines['bottom'].set_color(C_GRID); ax.spines['left'].set_color(C_GRID)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    
    # Lateral distribution  
    ax = axes[1]; ax.set_facecolor(C_CARD)
    ax.hist(f_lat / PATIENT_BW, bins=30, color=C_RED, alpha=0.7, edgecolor=C_GRID)
    ax.axvline(x=0, color=C_RED, linestyle='--', lw=2, label='Zero boundary')
    pct_ok = (total - lat_violations) / total * 100
    ax.set_title(f'Lateral Force (BW)\n{pct_ok:.0f}% compliant', color=C_TEXT, fontsize=11, fontweight='bold')
    ax.set_xlabel('Force (x BW)', color=C_TEXT)
    ax.legend(fontsize=8, framealpha=0.3)
    ax.tick_params(colors=C_SUB)
    ax.spines['bottom'].set_color(C_GRID); ax.spines['left'].set_color(C_GRID)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    
    # Medial/Lateral ratio over gait cycle (replaces hard-to-read scatter)
    ax = axes[2]; ax.set_facecolor(C_CARD)
    ratio = f_med[:n] / np.maximum(np.abs(f_lat[:n]), 1.0)
    ax.plot(gait_pct, ratio, color=C_PURPLE, lw=2)
    ax.axhline(y=0.6, color=C_GREEN, linestyle='--', alpha=0.5, label='Healthy range (0.4-0.8)')
    ax.axhline(y=0.4, color=C_GREEN, linestyle='--', alpha=0.5)
    ax.fill_between(gait_pct, 0.4, 0.8, alpha=0.05, color=C_GREEN)
    ax.set_xlabel('% Gait Cycle', color=C_TEXT)
    ax.set_ylabel('Medial / Lateral Ratio', color=C_TEXT)
    ax.set_title('M/L Force Ratio Over Gait Cycle', color=C_TEXT, fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, framealpha=0.3)
    ax.tick_params(colors=C_SUB)
    ax.spines['bottom'].set_color(C_GRID); ax.spines['left'].set_color(C_GRID)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.grid(color=C_GRID, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{OUT}/plots/physics_adherence.png', dpi=150, facecolor=C_BG, bbox_inches='tight')
    plt.close()
    
    # ── 4. Export predictions ──
    df_export = pd.DataFrame({
        '% Gait Cycle': gait_pct,
        'True_Fz_BW': fz_true_bw,
        'Predicted_Fz_BW': fz_pred_bw,
        'Predicted_Medial_BW': med_bw,
        'Predicted_Lateral_BW': lat_bw,
    })
    df_export.to_excel(f'{OUT}/patient_predictions_BW.xlsx', index=False)
    
    print(f"\n  All outputs saved to {OUT}/")
    return metrics


if __name__ == '__main__':
    model, dataset = train_overfit()
    metrics = evaluate_overfit(model, dataset)
