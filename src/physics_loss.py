import torch
import torch.nn as nn

class PhysicalConstraintsLoss(nn.Module):
    """
    Physics loss that computes MSE and adds constraints for the knee joint.
    """
    def __init__(self, lambda_physics=0.1, condyle_distance=0.04):
        super(PhysicalConstraintsLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.lambda_physics = lambda_physics
        # Approximated distance between medial and lateral condyles in meters
        self.condyle_distance = condyle_distance 
        
    def forward(self, y_pred, y_true):
        """
        y_pred shape: (batch_size, 6) -> Scaled [Fx, Fy, Fz, Tx, Ty, Tz]
        y_true shape: (batch_size, 6) -> Scaled [Fx, Fy, Fz, Tx, Ty, Tz]
        
        Physics rationale:
        The eTibia measures total axial force (Fz) and moments (e.g. Tx). 
        We assume a simple lever model where:
        Total Axial Force: Fz = F_medial + F_lateral
        Frontal Plane Moment: Tx = (F_medial - F_lateral) * (condyle_distance / 2)
        
        From these, we can algebraically solve for F_medial and F_lateral:
        F_medial = (Fz / 2) + (Tx / condyle_distance)
        F_lateral = (Fz / 2) - (Tx / condyle_distance)
        
        Contact forces must be compressive, therefore F_medial >= 0 and F_lateral >= 0.
        If they result in negative values, we penalize the network.
        
        Note: Since we are in the scaled space (RobustScaler), we ideally want this physical
        constraint in unscaled space. For a pure "soft constraint" in PyTorch without
        the scaler passed to the GPU, we penalize negativity of the implied medial/lateral ratio.
        """
        # 1. Main Data Loss (MSE on scaled predictions vs targets)
        data_loss = self.mse(y_pred, y_true)
        
        # 2. Physics Constraints (Soft limits)
        # Extract Fz (idx 2) and Tx (idx 3)
        # Note: Depending on the coordinate system, Ty might be the var/val moment. 
        # We will apply a generalized positivity constraint on the "implied" contact forces.
        fz_pred = y_pred[:, 2]
        tx_pred = y_pred[:, 3]
        
        # Implied medial and lateral forces (scaled space estimate)
        f_medial = (fz_pred / 2.0) + (tx_pred / self.condyle_distance)
        f_lateral = (fz_pred / 2.0) - (tx_pred / self.condyle_distance)
        
        # Physics loss: Penalize negative contact forces using ReLU
        # If f_medial < 0, relu(-f_medial) is > 0, producing a penalty
        penalty_medial = torch.mean(torch.relu(-f_medial)**2)
        penalty_lateral = torch.mean(torch.relu(-f_lateral)**2)
        
        physics_loss = penalty_medial + penalty_lateral
        
        total_loss = data_loss + self.lambda_physics * physics_loss
        
        return total_loss, data_loss, physics_loss
