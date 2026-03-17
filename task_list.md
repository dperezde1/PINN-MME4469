# PINN for JRF Prediction Task List

## 1. Project Setup & Data Loading
- [x] Create Python environment and install dependencies (PyTorch, Pandas, NumPy, Matplotlib, Scikit-learn, Seaborn).
- [x] Implement generic data parsers for SimTK synchronized data `.csv` files (Kinematics, GRF, EMG, eTibia).
- [x] **Data Preprocessing & Normalization:** Robust scaling (MinMax or Standard) of inputs and outputs, saving scalers for inference.
- [x] Implement PyTorch `Dataset` and `DataLoader` for batching (Input: Kinematics, GRF, EMG. Output: Knee Forces).

## 2. Physics-Informed Neural Network (PINN) Architecture
- [x] Define the base Neural Network (MLP) mapping inputs to JRF components.
- [x] Define the Physics-Informed Loss Function:
  - Data Loss (MSE between predicted JRF and eTibia true JRF).
  - Physics Loss (Ensuring medial + lateral = total axial force, and positive contact force constraints $F_{contact} \ge 0$).

## 3. Training Loop
- [x] Implement robust training loop (forward pass, loss compute, backprop) in PyTorch.
- [x] Track losses (Data loss vs Physics loss).
- [ ] Hyperparameter tuning (adjusting learning rate, batch size, and $\lambda$ weights for physics loss penalty).

## 4. Evaluation & Visualization (For Reporting)
- [x] Evaluate model on unseen trials.
- [x] **Correlation Analysis:** Generate R² correlation plots and residual scatter plots between predicted vs actual forces.
- [x] **Time-Series Plots:** Plot predicted vs actual Medial, Lateral, and Total Knee Contact Forces over the 1-100% gait cycle.
- [x] **Feature Importance / Weight Analysis:** Extract and visualize which inputs (kinematics vs GRF vs EMG) the model relies on most.
- [x] Export predictions in SimTK competition specified format (1-100% gait cycle Excel sheet).
