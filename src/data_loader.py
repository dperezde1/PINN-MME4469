import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler
import joblib

class SimTKDataset(Dataset):
    """
    PyTorch Dataset for SimTK Grand Challenge Joint Reaction Force Prediction.
    Loads kinematics, GRF, and EMG data as inputs, and eTibia forces/moments as targets.
    """
    def __init__(self, data_dir, trial_names, seq_length=None, is_train=True, scalers=None, scaler_save_dir='results'):
        self.data_dir = data_dir
        self.trial_names = trial_names
        self.seq_length = seq_length
        self.is_train = is_train
        self.scalers = scalers if scalers is not None else {}
        self.scaler_save_dir = scaler_save_dir
        
        self.inputs = []
        self.targets = []
        self.time_frames = []
        self.trial_indices = []
        
        self._load_data()
        
    def _load_data(self):
        emg_dir = os.path.join(self.data_dir, 'EMG Data')
        motion_dir = os.path.join(self.data_dir, 'Video Motion Data')
        etibia_dir = os.path.join(self.data_dir, 'eTibia Data')
        
        all_inputs = []
        all_targets = []
        
        for trial_idx, trial in enumerate(self.trial_names):
            # Define file paths
            emg_path = os.path.join(emg_dir, f"{trial}_emg.csv")
            grf_path = os.path.join(motion_dir, f"{trial}_grf.csv")
            traj_path = os.path.join(motion_dir, f"{trial}_trajectories.csv")
            force_path = os.path.join(etibia_dir, f"{trial}_knee_forces.csv")
            
            # Load CSVs
            # Note: We skip missing files (some trials might be incomplete)
            if not all(os.path.exists(p) for p in [emg_path, grf_path, traj_path, force_path]):
                print(f"Skipping {trial} due to missing files.")
                continue
                
            # Load CSVs with index_col=False to prevent panda from misinterpreting trailing commas
            df_emg = pd.read_csv(emg_path, index_col=False)
            df_grf = pd.read_csv(grf_path, index_col=False)
            df_traj = pd.read_csv(traj_path, index_col=False)
            df_force = pd.read_csv(force_path, index_col=False)
            
            # Standardize time columns for merging by renaming the FIRST dict match
            for df in [df_emg, df_grf, df_traj, df_force]:
                time_col = next((col for col in df.columns if 'time' in col.lower()), None)
                if time_col:
                    df.rename(columns={time_col: 'Time'}, inplace=True)
            
            # For merge_asof or simple merge, round time to 3 decimals to align 120Hz frames
            for df in [df_emg, df_grf, df_traj, df_force]:
                df['Time'] = df['Time'].round(3)
            
            # Drop unneeded columns like 'Frame' or 'GON' to avoid clutter
            if 'Frame' in df_traj.columns: df_traj.drop(columns=['Frame'], inplace=True)
            if 'Frame' in df_grf.columns: df_grf.drop(columns=['Frame'], inplace=True)
            
            # Merge dataframes on Time
            df_merged = df_emg.merge(df_grf, on='Time', how='inner')
            df_merged = df_merged.merge(df_traj, on='Time', how='inner')
            df_merged = df_merged.merge(df_force, on='Time', how='inner')
            
            # Filter columns: inputs vs targets
            # Targets: Fx, Fy, Fz, Tx, Ty, Tz
            target_cols = ['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz']
            df_targets = df_merged[target_cols].copy()
            
            # Inputs: everything else except 'Time' and possibly duplicate/unnecessary features
            input_cols = [c for c in df_merged.columns if c not in target_cols and c != 'Time' and c != 'GON' and c != 'GRFz_y']
            df_inputs = df_merged[input_cols].copy()
            
            # Fill NAs
            df_inputs.fillna(0, inplace=True)
            df_targets.fillna(0, inplace=True)
            
            # Store time for potential sequential slicing/reporting
            self.time_frames.append(df_merged['Time'].values)
            self.trial_indices.append(np.full(len(df_merged), trial_idx))
            
            all_inputs.append(df_inputs.values)
            all_targets.append(df_targets.values)
            
        print(f"Loaded {len(all_inputs)} valid trials.")
        
        # Concatenate all trials for scaler fitting
        concat_inputs = np.vstack(all_inputs)
        concat_targets = np.vstack(all_targets)
        
        if self.is_train:
            # Fit and save scalers
            self.scalers['inputs'] = RobustScaler().fit(concat_inputs)
            self.scalers['targets'] = RobustScaler().fit(concat_targets)
            os.makedirs(self.scaler_save_dir, exist_ok=True)
            joblib.dump(self.scalers['inputs'], os.path.join(self.scaler_save_dir, 'input_scaler.pkl'))
            joblib.dump(self.scalers['targets'], os.path.join(self.scaler_save_dir, 'target_scaler.pkl'))
            
        # Scale the data
        concat_inputs = self.scalers['inputs'].transform(concat_inputs)
        concat_targets = self.scalers['targets'].transform(concat_targets)
        
        # Convert to torch tensors
        self.inputs = torch.tensor(concat_inputs, dtype=torch.float32)
        self.targets = torch.tensor(concat_targets, dtype=torch.float32)
        self.time_frames = np.concatenate(self.time_frames)
        self.trial_indices = np.concatenate(self.trial_indices)
        
        self.feature_dim = self.inputs.shape[1]
        self.target_dim = self.targets.shape[1]
        self.feature_names = input_cols
        self.target_names = target_cols

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

def create_dataloaders(data_dir, trial_names, batch_size=64, validation_split=0.2):
    """
    Utility function to create train and validation DataLoaders.
    """
    # Simple split of trial names if there are enough, otherwise sequential split
    # For robust evaluation, it's better to leave out entire trials for validation
    num_trials = len(trial_names)
    val_size = max(1, int(num_trials * validation_split))
    
    train_trials = trial_names[:-val_size]
    val_trials = trial_names[-val_size:]
    
    print(f"Training on trials: {train_trials}")
    print(f"Validating on trials: {val_trials}")
    
    train_dataset = SimTKDataset(data_dir, train_trials, is_train=True)
    val_dataset = SimTKDataset(data_dir, val_trials, is_train=False, scalers=train_dataset.scalers)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, train_dataset.scalers

if __name__ == "__main__":
    # Test data loader
    data_dir = r"c:/Users/diego/OneDrive/Documents/MME4469/PINN/data/Overground Gait Trials"
    # Example trial names based on list_dir
    trial_names = ['DM_ngait_og1', 'DM_ngait_og2', 'DM_ngait_og3', 'DM_ngait_og4']
    train_loader, val_loader, scalers = create_dataloaders(data_dir, trial_names)
    
    for x, y in train_loader:
        print(f"Input batch shape: {x.shape}")
        print(f"Target batch shape: {y.shape}")
        break
