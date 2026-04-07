# PINN for JRF Prediction Walkthrough

We've successfully designed and trained a Physics-Informed Neural Network (PINN) to predict Knee Joint Reaction Forces (JRF) using the SimTK Grand Challenge dataset. Here is a review of what was accomplished and the resulting model performance.

## Project Accomplishments
- **Robust Data Pipeline**: Created `data_loader.py` to synchronize kinematics, Ground Reaction Forces (GRF), and Electromyography (EMG) signals based on precise timestamp alignment, handling trailing commas and scaling data with `RobustScaler`.
- **PINN Architecture**: Developed a PyTorch MLP (`model.py`) optimized with Exponential Linear Units (ELU) for smooth gradients.
- **Physical Constraints**: Formulated an algebraic physics loss (`physics_loss.py`) constraining the predicted medial and lateral forces. Given the total axial force ($F_z$) and frontal plane moments ($T_x$), the network is soft-penalized if it predicts anatomically impossible tension (forces must be $\ge 0$).
- **Training**: Executed PyTorch training looping (`train.py`) entirely on the CPU. The network successfully converged, driving down both data Mean Squared Error and the physical penalty loss.

## Model Performance & Evaluation

The model was trained on three gait trials and evaluated on a held-out validation trial. 

### Learning Curve
The optimizer successfully minimized the validation loss rapidly over 15 epochs.
![Learning Curve](file:///c:/Users/diego/.gemini/antigravity/brain/b8f1b4f7-fe7d-4951-bc4c-c0352812e7f4/learning_curve.png)

### JRF Correlation & Residuals
The true vs. predicted scatter plots for the three primary force vectors ($F_x$, $F_y$, $F_z$). 
![Correlation and Residuals](file:///c:/Users/diego/.gemini/antigravity/brain/b8f1b4f7-fe7d-4951-bc4c-c0352812e7f4/correlation_residuals.png)

### Gait Cycle Force Predictions
Here is how the prediction tracks across the stance and swing phases of the gait cycle. Notice the split between the Medial and Lateral compartments dictated by the Physics constraints.
![Gait Cycle Prediction](file:///c:/Users/diego/.gemini/antigravity/brain/b8f1b4f7-fe7d-4951-bc4c-c0352812e7f4/gait_cycle_forces.png)

### Feature Importance
A heuristic evaluation of the magnitude of the connection weights in the first layer. This demonstrates which sensor modalities the model relies on most heavily.
![Feature Importance](file:///c:/Users/diego/.gemini/antigravity/brain/b8f1b4f7-fe7d-4951-bc4c-c0352812e7f4/feature_importance.png)

## Future Execution
A `SimTK_Predictions.xlsx` file has been exported containing the 1-100% gait cycle predictions. 

Given the storage space constraints encountered during local `pip` installations, the project has been committed to a local Git repository, ignoring the massive `data/` and `venv/` directories. You can safely push this format to GitHub.
