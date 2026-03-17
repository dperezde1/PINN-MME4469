# Physics-Informed Neural Network (PINN) for Knee Joint Reaction Force Prediction

The goal is to develop a PyTorch-based PINN to predict in vivo knee contact forces (JRF) during walking using the SimTK Grand Challenge dataset (Synchronized motion, GRF, EMG, and eTibia data). 

## User Review Required

> [!IMPORTANT]
> **Physics Loss Formulation:** A standard neural network is trained only on data (`Loss = MSE(Y_pred - Y_true)`). A PINN adds a physical constraint penalty (`Loss = MSE + lambda * Physics_Loss`). 
> To calculate a rigorous `Physics_Loss` for joint reaction forces from scratch requires a full musculoskeletal rigid body inverse dynamics solver (like OpenSim) embedded inside the PyTorch graph, which is computationally immense. 
> 
> **My Proposal:** We will implement a simplified Physics Loss. For example, ensuring that the predicted medial and lateral forces sum up correctly to the predicted total axial force, and placing soft constraints on the direction/boundaries of the forces (e.g., contact forces must be compressive, so $F_{contact} \ge 0$). Alternatively, if you have a specific equation of motion for your PINN in mind, please let me know!

> [!TIP]
> **Tooling Formulation:** I will set this up as a Python project using Object-Oriented Programming (OOP) principles so you can easily scale it. I'll include `requirements.txt` to install necessary libs (PyTorch, pandas).

## Proposed Changes

### Project Structure Setup
#### [NEW] `c:\Users\diego\OneDrive\Documents\MME4469\PINN\requirements.txt`
Dependencies (PyTorch, Pandas, Scipy, Matplotlib).
#### [NEW] `c:\Users\diego\OneDrive\Documents\MME4469\PINN\src\data_loader.py`
A custom PyTorch Dataset to parse the CSV files from `Overground Gait Trials` and `Treadmill Gait Trials` and handle alignment in time.
#### [NEW] `c:\Users\diego\OneDrive\Documents\MME4469\PINN\src\model.py`
The Neural Network PyTorch `nn.Module`.
#### [NEW] `c:\Users\diego\OneDrive\Documents\MME4469\PINN\src\physics_loss.py`
The custom loss function enforcing the physics penalties.
#### [NEW] `c:\Users\diego\OneDrive\Documents\MME4469\PINN\src\train.py`
The main training loop.
#### [NEW] `c:\Users\diego\OneDrive\Documents\MME4469\PINN\notebooks\1_Exploration_and_Visualization.ipynb`
A Jupyter Notebook for exploring the loaded data and visualizing the final model predictions against the true eTibia contact forces.

## Verification Plan

### Automated Tests
* We will verify the `data_loader.py` correctly reads the `.csv` files and generates PyTorch tensors of matching shapes without NaNs.
* Verification of `physics_loss.py` gradients (ensuring `torch.autograd` successfully backpropagates through the physics penalties).

### Manual Verification
* Generating comparison plots in the Jupyter Notebook to visually ensure the PINN successfully predicts Medial and Lateral contact forces over the 1-100% gait cycle with reasonable accuracy relative to the true `eTibia` targets.
