import os
import torch
import torch.optim as optim
from data_loader import create_dataloaders
from model import KneePINN
from physics_loss import PhysicalConstraintsLoss
import matplotlib.pyplot as plt

def train_model():
    data_dir = r"c:/Users/bennn/localSchool/PINN/PINN-MME4469/data/Overground Gait Trials"
    
    # Example trials from eTibia data listing
    trial_names = ['DM_ngait_og1', 'DM_ngait_og2', 'DM_ngait_og3', 'DM_ngait_og4']
    
    epochs = 300 
    batch_size = 32
    learning_rate = 1e-2
    
    # User constraint: No CUDA
    device = torch.device('cpu') 
    
    print("Initializing Data Loaders...")
    train_loader, val_loader, scalers = create_dataloaders(data_dir, trial_names, batch_size=batch_size)
    
    if len(train_loader.dataset) == 0:
        print("Error: Train dataset is empty. Check data paths/alignments.")
        return
        
    input_dim = train_loader.dataset.feature_dim
    output_dim = train_loader.dataset.target_dim
    
    print(f"Features: {input_dim}, Targets: {output_dim}")
    
    model = KneePINN(input_dim=input_dim, output_dim=output_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-5, verbose=True)
    criterion = PhysicalConstraintsLoss(lambda_physics=0.05)
    
    train_history, val_history = [], []
    best_val_loss = float('inf')
    os.makedirs('results', exist_ok=True)
    model_save_path = 'results/best_pinn_model.pth'
    
    early_stop_counter = 0
    patience = 50
    
    print(f"Entering training loop for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            preds = model(inputs)
            total_loss, data_loss, phys_loss = criterion(preds, targets)
            
            total_loss.backward()
            optimizer.step()
            
            running_train_loss += total_loss.item()
            
        avg_train_loss = running_train_loss / len(train_loader)
        train_history.append(avg_train_loss)
        
        # Validation Phase
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                preds = model(inputs)
                total_loss, data_loss, phys_loss = criterion(preds, targets)
                running_val_loss += total_loss.item()
                
        avg_val_loss = running_val_loss / max(1, len(val_loader))
        val_history.append(avg_val_loss)
        
        if avg_val_loss < best_val_loss and len(val_loader) > 0:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_save_path)
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            
        scheduler.step(avg_val_loss)
            
        if epoch % 2 == 0 or epoch == epochs - 1:
            print(f"Epoch [{epoch}/{epochs}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
            
        if early_stop_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch}")
            break
            
    print(f"Training Complete! Best Val Loss: {best_val_loss:.4f}")
    
    # Save learning curve
    plt.figure(figsize=(10,6))
    plt.plot(train_history, label='Train Loss', color='blue')
    if len(val_loader) > 0:
        plt.plot(val_history, label='Validation Loss', color='orange')
    plt.title('PINN Learning Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('results/plots/learning_curve.png')
    print("Learning curve saved to results/plots/learning_curve.png")

if __name__ == "__main__":
    train_model()
