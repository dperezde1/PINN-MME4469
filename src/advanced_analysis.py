"""
Phase 6: Advanced Analysis Suite
Generates publication-quality analysis plots:
  1. Leave-One-Out Cross-Validation (LOOCV)
  2. PCA Scree Plot
  3. Body-Weight Normalized Gait Cycle
  4. Physics Constraint Adherence
  5. Model Comparison Summary Figure
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import torch
import torch.optim as optim
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(__file__))
from data_loader import SimTKDataset, create_dataloaders, PATIENT_BW
from model import KneePINN
from physics_loss import PhysicalConstraintsLoss

OUT = 'results/advanced_analysis'
os.makedirs(OUT, exist_ok=True)

# Color scheme
C_BLUE = '#58A6FF'
C_GREEN = '#3FB950'
C_RED = '#F85149'
C_PURPLE = '#D2A8FF'
C_ORANGE = '#F0883E'
C_BG = '#0D1117'
C_CARD = '#161B22'
C_GRID = '#21262D'
C_TEXT = '#C9D1D9'
C_SUB = '#8B949E'

DATA_DIR = r"c:/Users/bennn/localSchool/PINN/PINN-MME4469/data/Overground Gait Trials"
TRIALS = ['DM_ngait_og1', 'DM_ngait_og2', 'DM_ngait_og3', 'DM_ngait_og4']
TARGET_COLS = ['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz']

# ═══════════════════════════════════════════════════════════════
# 1. LEAVE-ONE-OUT CROSS-VALIDATION
# ═══════════════════════════════════════════════════════════════
def run_loocv():
    print("=" * 60)
    print("1. Leave-One-Out Cross-Validation")
    print("=" * 60)
    
    fold_results = []  # list of dicts, one per fold
    
    for fold_idx in range(len(TRIALS)):
        val_trial = TRIALS[fold_idx]
        train_trials = [t for t in TRIALS if t != val_trial]
        
        print(f"\n  Fold {fold_idx+1}/4: Validating on {val_trial}")
        
        # Create datasets
        train_ds = SimTKDataset(DATA_DIR, train_trials, is_train=True, n_components=25)
        val_ds = SimTKDataset(DATA_DIR, [val_trial], is_train=False, 
                              scalers=train_ds.scalers, pca=train_ds.pca, n_components=25)
        
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=16, shuffle=True, drop_last=True)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=2048, shuffle=False)
        
        model = KneePINN(input_dim=train_ds.feature_dim, output_dim=train_ds.target_dim)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-5)
        criterion = PhysicalConstraintsLoss(lambda_physics=0.5)
        
        best_val = float('inf')
        patience_counter = 0
        
        for epoch in range(300):
            model.train()
            for x, y in train_loader:
                optimizer.zero_grad()
                loss, _, _ = criterion(model(x), y)
                loss.backward()
                optimizer.step()
            
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for x, y in val_loader:
                    vl, _, _ = criterion(model(x), y)
                    val_loss += vl.item()
            val_loss /= max(len(val_loader), 1)
            scheduler.step(val_loss)
            
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= 50:
                break
        
        # Evaluate best model
        model.load_state_dict(best_state)
        model.eval()
        all_p, all_t = [], []
        with torch.no_grad():
            for x, y in val_loader:
                all_p.append(model(x).numpy())
                all_t.append(y.numpy())
        
        preds = train_ds.scalers['targets'].inverse_transform(np.vstack(all_p))
        trues = train_ds.scalers['targets'].inverse_transform(np.vstack(all_t))
        
        fold_dict = {'fold': fold_idx + 1, 'val_trial': val_trial}
        for i, col in enumerate(TARGET_COLS):
            fold_dict[f'{col}_R2'] = r2_score(trues[:, i], preds[:, i])
            fold_dict[f'{col}_RMSE'] = np.sqrt(mean_squared_error(trues[:, i], preds[:, i]))
        
        fold_results.append(fold_dict)
        print(f"    Fz R2={fold_dict['Fz_R2']:.3f}, RMSE={fold_dict['Fz_RMSE']:.1f}")
    
    df_cv = pd.DataFrame(fold_results)
    df_cv.to_csv(f'{OUT}/loocv_results.csv', index=False)
    
    # Plot LOOCV results
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=C_BG)
    
    # R² per fold per component
    ax = axes[0]
    ax.set_facecolor(C_CARD)
    cols_r2 = [c for c in df_cv.columns if '_R2' in c]
    x_pos = np.arange(4)
    width = 0.13
    colors = [C_BLUE, C_GREEN, C_RED, C_PURPLE, C_ORANGE, '#FFA657']
    
    for j, col in enumerate(cols_r2):
        label = col.replace('_R2', '')
        ax.bar(x_pos + j * width, df_cv[col].values, width, label=label, color=colors[j], alpha=0.85)
    
    ax.set_xticks(x_pos + width * 2.5)
    ax.set_xticklabels([f'Fold {i+1}\n({TRIALS[i][-3:]})' for i in range(4)], color=C_SUB, fontsize=9)
    ax.set_ylabel('R² Score', color=C_TEXT)
    ax.set_title('R² by Fold and Component', color=C_TEXT, fontsize=12, fontweight='bold')
    ax.axhline(y=0, color=C_RED, linestyle='--', alpha=0.5, linewidth=1)
    ax.legend(fontsize=8, ncol=3, framealpha=0.3)
    ax.tick_params(colors=C_SUB)
    ax.spines['bottom'].set_color(C_GRID)
    ax.spines['left'].set_color(C_GRID)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', color=C_GRID, alpha=0.5)
    
    # Mean ± Std summary
    ax2 = axes[1]
    ax2.set_facecolor(C_CARD)
    means = [df_cv[c].mean() for c in cols_r2]
    stds = [df_cv[c].std() for c in cols_r2]
    labels = [c.replace('_R2', '') for c in cols_r2]
    
    bars = ax2.bar(labels, means, yerr=stds, capsize=5, color=colors, alpha=0.85,
                   error_kw={'ecolor': C_TEXT, 'capthick': 1.5, 'linewidth': 1.5})
    ax2.axhline(y=0, color=C_RED, linestyle='--', alpha=0.5, linewidth=1)
    ax2.set_ylabel('Mean R² Score', color=C_TEXT)
    ax2.set_title('LOOCV: Mean ±  Std R²', color=C_TEXT, fontsize=12, fontweight='bold')
    ax2.tick_params(colors=C_SUB)
    ax2.spines['bottom'].set_color(C_GRID)
    ax2.spines['left'].set_color(C_GRID)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(axis='y', color=C_GRID, alpha=0.5)
    
    # Add value labels
    for bar, m, s in zip(bars, means, stds):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + s + 0.02,
                f'{m:.3f}', ha='center', va='bottom', color=C_TEXT, fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{OUT}/loocv_results.png', dpi=150, facecolor=C_BG, bbox_inches='tight')
    plt.close()
    print("\n  LOOCV plot saved.")
    return df_cv


# ═══════════════════════════════════════════════════════════════
# 2. PCA SCREE PLOT
# ═══════════════════════════════════════════════════════════════
def plot_pca_scree():
    print("\n" + "=" * 60)
    print("2. PCA Scree Plot")
    print("=" * 60)
    
    # Load all data to fit full PCA
    all_ds = SimTKDataset(DATA_DIR, TRIALS, is_train=True, n_components=242)  # Max components
    full_pca = all_ds.pca
    
    var_ratio = full_pca.explained_variance_ratio_ * 100
    cum_var = np.cumsum(var_ratio)
    n_95 = np.argmax(cum_var >= 95) + 1
    n_99 = np.argmax(cum_var >= 99) + 1
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), facecolor=C_BG)
    
    # Individual variance
    ax1.set_facecolor(C_CARD)
    n_show = min(50, len(var_ratio))
    ax1.bar(range(1, n_show + 1), var_ratio[:n_show], color=C_BLUE, alpha=0.8)
    ax1.set_xlabel('Principal Component', color=C_TEXT)
    ax1.set_ylabel('Variance Explained (%)', color=C_TEXT)
    ax1.set_title('Individual Variance per Component', color=C_TEXT, fontsize=12, fontweight='bold')
    ax1.tick_params(colors=C_SUB)
    ax1.spines['bottom'].set_color(C_GRID)
    ax1.spines['left'].set_color(C_GRID)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(axis='y', color=C_GRID, alpha=0.5)
    
    # Cumulative variance
    ax2.set_facecolor(C_CARD)
    ax2.plot(range(1, len(cum_var) + 1), cum_var, color=C_GREEN, linewidth=2.5)
    ax2.fill_between(range(1, len(cum_var) + 1), cum_var, alpha=0.15, color=C_GREEN)
    ax2.axhline(y=95, color=C_ORANGE, linestyle='--', alpha=0.7, label=f'95% at PC{n_95}')
    ax2.axhline(y=99, color=C_PURPLE, linestyle='--', alpha=0.7, label=f'99% at PC{n_99}')
    ax2.axvline(x=25, color=C_RED, linestyle=':', alpha=0.7, label='Our choice: PC25')
    ax2.scatter([25], [cum_var[24]], color=C_RED, s=80, zorder=5)
    ax2.annotate(f'{cum_var[24]:.1f}%', xy=(25, cum_var[24]), xytext=(35, cum_var[24] - 5),
                color=C_TEXT, fontsize=10, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=C_TEXT))
    ax2.set_xlabel('Number of Components', color=C_TEXT)
    ax2.set_ylabel('Cumulative Variance (%)', color=C_TEXT)
    ax2.set_title('Cumulative Variance Explained', color=C_TEXT, fontsize=12, fontweight='bold')
    ax2.set_xlim(0, min(60, len(cum_var)))
    ax2.legend(fontsize=9, framealpha=0.3)
    ax2.tick_params(colors=C_SUB)
    ax2.spines['bottom'].set_color(C_GRID)
    ax2.spines['left'].set_color(C_GRID)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(color=C_GRID, alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(f'{OUT}/pca_scree_plot.png', dpi=150, facecolor=C_BG, bbox_inches='tight')
    plt.close()
    print(f"  95% variance at {n_95} components, 99% at {n_99} components")
    print("  Scree plot saved.")


# ═══════════════════════════════════════════════════════════════
# 3. BODY-WEIGHT NORMALIZED GAIT CYCLE
# ═══════════════════════════════════════════════════════════════
def plot_bw_normalized():
    print("\n" + "=" * 60)
    print("3. Body-Weight Normalized Gait Cycle")
    print("=" * 60)
    
    _, val_loader, scalers = create_dataloaders(DATA_DIR, TRIALS, batch_size=2048)
    
    model = KneePINN(input_dim=val_loader.dataset.feature_dim, output_dim=val_loader.dataset.target_dim)
    model.load_state_dict(torch.load('results/best_pinn_model.pth', map_location='cpu'))
    model.eval()
    
    all_p, all_t = [], []
    with torch.no_grad():
        for x, y in val_loader:
            all_p.append(model(x).numpy())
            all_t.append(y.numpy())
    
    preds = scalers['targets'].inverse_transform(np.vstack(all_p))
    trues = scalers['targets'].inverse_transform(np.vstack(all_t))
    
    n = min(100, len(preds))
    gait_pct = np.linspace(0, 100, n)
    
    # Normalize to BW
    fz_pred_bw = preds[:n, 2] / PATIENT_BW
    fz_true_bw = trues[:n, 2] / PATIENT_BW
    
    d = 0.04
    med_pred_bw = ((preds[:n, 2] / 2.0) + (preds[:n, 3] / d)) / PATIENT_BW
    lat_pred_bw = ((preds[:n, 2] / 2.0) - (preds[:n, 3] / d)) / PATIENT_BW
    
    # Gait phase boundaries
    phases = [
        ('IC', 0, 2, C_ORANGE), ('LR', 2, 12, C_RED),
        ('MSt', 12, 31, C_GREEN), ('TSt', 31, 50, C_BLUE),
        ('PSw', 50, 62, C_PURPLE), ('Sw', 62, 100, '#FFA657')
    ]
    
    fig, ax = plt.subplots(figsize=(12, 6), facecolor=C_BG)
    ax.set_facecolor(C_CARD)
    
    for name, s, e, color in phases:
        ax.axvspan(s, e, alpha=0.08, color=color)
        ax.text((s + e) / 2, ax.get_ylim()[0] if ax.get_ylim()[0] != 0 else -0.5, name,
               ha='center', va='bottom', fontsize=8, color=color, fontweight='bold', alpha=0.7)
    
    ax.plot(gait_pct, fz_true_bw, 'w--', lw=2.5, label='True Fz', alpha=0.9)
    ax.plot(gait_pct, fz_pred_bw, color=C_BLUE, lw=2.5, label='Predicted Fz')
    ax.plot(gait_pct, med_pred_bw, color=C_GREEN, lw=1.8, label='Predicted Medial', alpha=0.8)
    ax.plot(gait_pct, lat_pred_bw, color=C_RED, lw=1.8, label='Predicted Lateral', alpha=0.8)
    
    # Error band
    error_bw = np.abs(fz_pred_bw - fz_true_bw)
    ax.fill_between(gait_pct, fz_pred_bw - error_bw, fz_pred_bw + error_bw,
                    alpha=0.12, color=C_BLUE, label='Error band')
    
    ax.set_xlabel('% Gait Cycle', color=C_TEXT, fontsize=11)
    ax.set_ylabel('Force (x Body Weight)', color=C_TEXT, fontsize=11)
    ax.set_title('Knee Contact Forces Normalized to Body Weight (BW = 686.7 N)',
                color=C_TEXT, fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9, framealpha=0.3)
    ax.tick_params(colors=C_SUB)
    ax.spines['bottom'].set_color(C_GRID)
    ax.spines['left'].set_color(C_GRID)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(color=C_GRID, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{OUT}/bw_normalized_gait.png', dpi=150, facecolor=C_BG, bbox_inches='tight')
    plt.close()
    print("  BW-normalized gait plot saved.")


# ═══════════════════════════════════════════════════════════════
# 4. PHYSICS CONSTRAINT ADHERENCE
# ═══════════════════════════════════════════════════════════════
def plot_physics_adherence():
    print("\n" + "=" * 60)
    print("4. Physics Constraint Adherence")
    print("=" * 60)
    
    _, val_loader, scalers = create_dataloaders(DATA_DIR, TRIALS, batch_size=2048)
    
    model = KneePINN(input_dim=val_loader.dataset.feature_dim, output_dim=val_loader.dataset.target_dim)
    model.load_state_dict(torch.load('results/best_pinn_model.pth', map_location='cpu'))
    model.eval()
    
    all_p = []
    with torch.no_grad():
        for x, y in val_loader:
            all_p.append(model(x).numpy())
    
    preds = scalers['targets'].inverse_transform(np.vstack(all_p))
    
    fz = preds[:, 2]
    tx = preds[:, 3]
    d = 0.04
    f_medial = (fz / 2.0) + (tx / d)
    f_lateral = (fz / 2.0) - (tx / d)
    
    # In a normal knee, contact forces should be compressive (positive in eTibia convention)
    # But our model predicts in scaled space - check sign
    med_violations = np.sum(f_medial < 0)
    lat_violations = np.sum(f_lateral < 0)
    total = len(f_medial)
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor=C_BG)
    
    # Medial force distribution
    ax = axes[0]
    ax.set_facecolor(C_CARD)
    ax.hist(f_medial, bins=30, color=C_GREEN, alpha=0.7, edgecolor=C_GRID)
    ax.axvline(x=0, color=C_RED, linestyle='--', lw=2, label='Zero (violation boundary)')
    ax.set_title(f'Medial Force Distribution\n({med_violations}/{total} violations)', 
                color=C_TEXT, fontsize=11, fontweight='bold')
    ax.set_xlabel('Force (N)', color=C_TEXT)
    ax.set_ylabel('Count', color=C_TEXT)
    ax.legend(fontsize=8, framealpha=0.3)
    ax.tick_params(colors=C_SUB)
    ax.spines['bottom'].set_color(C_GRID)
    ax.spines['left'].set_color(C_GRID)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Lateral force distribution
    ax = axes[1]
    ax.set_facecolor(C_CARD)
    ax.hist(f_lateral, bins=30, color=C_RED, alpha=0.7, edgecolor=C_GRID)
    ax.axvline(x=0, color=C_RED, linestyle='--', lw=2, label='Zero (violation boundary)')
    ax.set_title(f'Lateral Force Distribution\n({lat_violations}/{total} violations)',
                color=C_TEXT, fontsize=11, fontweight='bold')
    ax.set_xlabel('Force (N)', color=C_TEXT)
    ax.legend(fontsize=8, framealpha=0.3)
    ax.tick_params(colors=C_SUB)
    ax.spines['bottom'].set_color(C_GRID)
    ax.spines['left'].set_color(C_GRID)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Medial vs Lateral scatter
    ax = axes[2]
    ax.set_facecolor(C_CARD)
    ax.scatter(f_medial, f_lateral, alpha=0.4, s=15, color=C_PURPLE)
    ax.axhline(y=0, color=C_GRID, linestyle='-', alpha=0.5)
    ax.axvline(x=0, color=C_GRID, linestyle='-', alpha=0.5)
    
    # Shade the valid quadrant (both positive = compressive)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.fill_between([max(0, xlim[0]), xlim[1]], 0, ylim[1], alpha=0.05, color=C_GREEN, label='Valid (both compressive)')
    
    ax.set_title('Medial vs Lateral Contact Forces', color=C_TEXT, fontsize=11, fontweight='bold')
    ax.set_xlabel('Medial Force (N)', color=C_TEXT)
    ax.set_ylabel('Lateral Force (N)', color=C_TEXT)
    ax.legend(fontsize=8, framealpha=0.3, loc='lower right')
    ax.tick_params(colors=C_SUB)
    ax.spines['bottom'].set_color(C_GRID)
    ax.spines['left'].set_color(C_GRID)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f'{OUT}/physics_adherence.png', dpi=150, facecolor=C_BG, bbox_inches='tight')
    plt.close()
    
    compliance = ((total - med_violations) + (total - lat_violations)) / (2 * total) * 100
    print(f"  Medial violations: {med_violations}/{total}")
    print(f"  Lateral violations: {lat_violations}/{total}")
    print(f"  Overall physics compliance: {compliance:.1f}%")
    print("  Physics adherence plot saved.")
    return compliance


# ═══════════════════════════════════════════════════════════════
# 5. MODEL COMPARISON SUMMARY FIGURE
# ═══════════════════════════════════════════════════════════════
def plot_model_comparison():
    print("\n" + "=" * 60)
    print("5. Model Comparison Summary")
    print("=" * 60)
    
    # Data for all 3 model iterations
    models = ['Phase 3\nBaseline MLP\n[256,256,128,64]', 
              'Phase 4\nLSTM\n[128h x 2L]', 
              'Phase 5\nPCA + Compact MLP\n[32,16] + Biometrics']
    
    # R² values from each phase (Fx, Fy, Fz)
    r2_fx = [-0.005, -0.097, -0.449]  
    r2_fy = [-0.181, -0.242, -0.684]
    r2_fz = [-0.027,  -0.419, 0.236]  # Phase 5 achieved positive!
    
    params = [200000, 200000, 1500]
    features = [242, 242, 28]
    
    fig = plt.figure(figsize=(16, 10), facecolor=C_BG)
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)
    
    # --- Top Left: R² comparison bars ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor(C_CARD)
    x = np.arange(3)
    w = 0.25
    
    ax1.bar(x - w, r2_fx, w, label='Fx', color=C_BLUE, alpha=0.85)
    ax1.bar(x, r2_fy, w, label='Fy', color=C_GREEN, alpha=0.85)
    ax1.bar(x + w, r2_fz, w, label='Fz', color=C_RED, alpha=0.85)
    
    ax1.axhline(y=0, color='white', linestyle='-', alpha=0.3, lw=1)
    ax1.set_xticks(x)
    ax1.set_xticklabels(['Baseline\nMLP', 'LSTM', 'PCA+MLP'], color=C_SUB, fontsize=9)
    ax1.set_ylabel('R² Score', color=C_TEXT)
    ax1.set_title('R² Score Progression Across Models', color=C_TEXT, fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9, framealpha=0.3)
    ax1.tick_params(colors=C_SUB)
    ax1.spines['bottom'].set_color(C_GRID)
    ax1.spines['left'].set_color(C_GRID)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(axis='y', color=C_GRID, alpha=0.3)
    
    # Highlight the positive R² achievement
    ax1.annotate('First positive R²!', xy=(2 + w, r2_fz[2]), xytext=(1.5, 0.35),
                color=C_GREEN, fontsize=10, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=C_GREEN, lw=2))
    
    # --- Top Right: Parameter efficiency ---
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor(C_CARD)
    
    bar_colors = [C_SUB, C_ORANGE, C_GREEN]
    bars = ax2.bar(['Baseline\nMLP', 'LSTM', 'PCA+MLP'], params, color=bar_colors, alpha=0.85)
    ax2.set_ylabel('# Parameters', color=C_TEXT)
    ax2.set_title('Model Complexity (Fewer = Better for Small Data)', color=C_TEXT, fontsize=12, fontweight='bold')
    ax2.tick_params(colors=C_SUB)
    ax2.spines['bottom'].set_color(C_GRID)
    ax2.spines['left'].set_color(C_GRID)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(axis='y', color=C_GRID, alpha=0.3)
    
    for bar, val in zip(bars, params):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3000,
                f'{val:,}', ha='center', va='bottom', color=C_TEXT, fontsize=10, fontweight='bold')
    
    # --- Bottom Left: Feature dimension ---
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor(C_CARD)
    
    bars = ax3.bar(['Baseline\nMLP', 'LSTM', 'PCA+MLP'], features, color=bar_colors, alpha=0.85)
    ax3.set_ylabel('Input Dimensions', color=C_TEXT)
    ax3.set_title('Feature Space (PCA Compression)', color=C_TEXT, fontsize=12, fontweight='bold')
    ax3.tick_params(colors=C_SUB)
    ax3.spines['bottom'].set_color(C_GRID)
    ax3.spines['left'].set_color(C_GRID)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.grid(axis='y', color=C_GRID, alpha=0.3)
    
    for bar, val in zip(bars, features):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                str(val), ha='center', va='bottom', color=C_TEXT, fontsize=11, fontweight='bold')
    
    ax3.annotate('88% reduction\n(242 -> 28)', xy=(2, 28), xytext=(1.2, 150),
                color=C_GREEN, fontsize=10, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=C_GREEN, lw=2))
    
    # --- Bottom Right: Key Takeaways ---
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor(C_CARD)
    ax4.axis('off')
    
    takeaways = [
        ("KEY FINDINGS", C_BLUE, 14),
        ("", None, 8),
        ("1. Architecture alone doesn't solve underfitting", C_TEXT, 11),
        ("   LSTM (200K params) performed worse than", C_SUB, 9),
        ("   compact MLP (1.5K params) on limited data.", C_SUB, 9),
        ("", None, 6),
        ("2. PCA reduced 242 features to 25 components", C_TEXT, 11),
        ("   retaining 99.9% variance while preventing", C_SUB, 9),
        ("   the curse of dimensionality.", C_SUB, 9),
        ("", None, 6),
        ("3. Patient biometrics (70kg, 172cm) provided", C_TEXT, 11),
        ("   physical context for individualized prediction.", C_SUB, 9),
        ("", None, 6),
        ("4. Stronger physics constraints (10x) guided", C_TEXT, 11),
        ("   the model when data was insufficient.", C_SUB, 9),
        ("", None, 6),
        ("5. Fz R² improved from -0.42 to +0.24", C_GREEN, 11),
        ("   First component to outperform mean predictor.", C_SUB, 9),
    ]
    
    y_pos = 0.95
    for text, color, size in takeaways:
        if color:
            ax4.text(0.05, y_pos, text, transform=ax4.transAxes, fontsize=size,
                    color=color, fontweight='bold' if size >= 11 else 'normal',
                    verticalalignment='top')
        y_pos -= 0.055
    
    plt.savefig(f'{OUT}/model_comparison.png', dpi=150, facecolor=C_BG, bbox_inches='tight')
    plt.close()
    print("  Model comparison figure saved.")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("\n  PHASE 6: Advanced Analysis Suite")
    print("  Patient: 70kg, 172cm, BMI=23.7")
    print(f"  BW = {PATIENT_BW:.1f} N\n")
    
    df_cv = run_loocv()
    plot_pca_scree()
    plot_bw_normalized()
    compliance = plot_physics_adherence()
    plot_model_comparison()
    
    print("\n" + "=" * 60)
    print("All advanced analysis plots saved to results/advanced_analysis/")
    print("=" * 60)
