import os
import pandas as pd

data_dir = r"c:/Users/diego/OneDrive/Documents/MME4469/PINN/data/Overground Gait Trials"
trial = "DM_ngait_og1"

emg_dir = os.path.join(data_dir, 'EMG Data')
motion_dir = os.path.join(data_dir, 'Video Motion Data')
etibia_dir = os.path.join(data_dir, 'eTibia Data')

emg_path = os.path.join(emg_dir, f"{trial}_emg.csv")
grf_path = os.path.join(motion_dir, f"{trial}_grf.csv")
traj_path = os.path.join(motion_dir, f"{trial}_trajectories.csv")
force_path = os.path.join(etibia_dir, f"{trial}_knee_forces.csv")

print("Files exist:", [os.path.exists(p) for p in [emg_path, grf_path, traj_path, force_path]])

df_emg = pd.read_csv(emg_path)
df_grf = pd.read_csv(grf_path)
df_traj = pd.read_csv(traj_path)
df_force = pd.read_csv(force_path)

for name, df in zip(['EMG', 'GRF', 'Traj', 'Force'], [df_emg, df_grf, df_traj, df_force]):
    time_cols = [c for c in df.columns if 'time' in c.lower()]
    if time_cols:
        tcol = time_cols[0]
        print(f"{name} bounds [{df[tcol].min()}, {df[tcol].max()}], count: {len(df)}")
    else:
        print(f"{name} has NO time column!")

df_emg.rename(columns={col: 'Time' for col in df_emg.columns if 'time' in col.lower()}, inplace=True)
df_grf.rename(columns={col: 'Time' for col in df_grf.columns if 'time' in col.lower()}, inplace=True)
df_traj.rename(columns={col: 'Time' for col in df_traj.columns if 'time' in col.lower()}, inplace=True)
df_force.rename(columns={col: 'Time' for col in df_force.columns if 'time' in col.lower()}, inplace=True)

for df in [df_emg, df_grf, df_traj, df_force]:
    df['Time'] = df['Time'].round(3)

print("EMG sample:", df_emg['Time'].head().values)
print("GRF sample:", df_grf['Time'].head().values)
print("Traj sample:", df_traj['Time'].head().values)
print("Force sample:", df_force['Time'].head().values)

df_merged = df_emg.merge(df_grf, on='Time', how='inner')
print("After EMG+GRF:", len(df_merged))
df_merged = df_merged.merge(df_traj, on='Time', how='inner')
print("After +Traj:", len(df_merged))
df_merged = df_merged.merge(df_force, on='Time', how='inner')
print("After +Force:", len(df_merged))
