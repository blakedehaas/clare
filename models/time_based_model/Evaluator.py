# Evaluator.py

import torch
import numpy as np
import shap

class Evaluator:
    def __init__(self, model, criterion, device):
        """
        Initializes the Evaluator.
        
        Args:
            model (torch.nn.Module): The trained model.
            criterion (torch.nn.Module): Loss function.
            device (torch.device): Device to run evaluations on.
        """
        self.model = model
        self.criterion = criterion
        self.device = device

    def evaluate_metrics(self, data_loader):
        """
        Evaluates the model on the provided data loader and computes the average loss.
        
        Args:
            data_loader (DataLoader): DataLoader for evaluation data.
        
        Returns:
            float: Average loss over the dataset.
        """
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for x, y in data_loader:
                x, y = x.to(self.device), y.to(self.device)
                y_pred = self.model(x)
                loss = self.criterion(y_pred, y)
                total_loss += loss.item() * x.size(0)
        avg_loss = total_loss / len(data_loader.dataset)
        return avg_loss

    def residual_analysis(self, data_loader):
        """
        Performs residual analysis by computing the difference between predictions and true values.
        
        Args:
            data_loader (DataLoader): DataLoader for evaluation data.
        
        Returns:
            np.ndarray: Array of residuals.
        """
        self.model.eval()
        residuals = []
        with torch.no_grad():
            for x, y in data_loader:
                x, y = x.to(self.device), y.to(self.device)
                y_pred = self.model(x)
                residual = y_pred - y
                residuals.append(residual.cpu().numpy())
        residuals = np.concatenate(residuals, axis=0)
        return residuals

    def compute_feature_importance(self, sample_data):
        """
        Computes feature importance using SHAP for the given sample data.
        
        Args:
            sample_data (torch.Tensor): Sample data for SHAP analysis.
        
        Returns:
            shap.Explanation: SHAP values.
        """
        # Ensure the model is on CPU for SHAP
        model_cpu = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model
        model_cpu.to('cpu')
        model_cpu.eval()
    
        # Convert sample data to numpy
        sample_data_np = sample_data.detach().cpu().numpy()
        
        # Reshape sample_data_np to 2D if it's 3D (e.g., [batch_size, seq_length, num_features])
        if sample_data_np.ndim > 2:
            sample_data_np = sample_data_np.reshape(sample_data_np.shape[0], -1)
    
        # Define a prediction function for SHAP
        def model_predict(data):
            data_tensor = torch.from_numpy(data).float().to('cpu')
            with torch.no_grad():
                outputs = model_cpu(data_tensor)
            return outputs.detach().cpu().numpy()
    
        # Use a subset of the data as background for KernelExplainer
        background = sample_data_np[:100] if sample_data_np.shape[0] >= 100 else sample_data_np
    
        # Initialize SHAP KernelExplainer
        explainer = shap.KernelExplainer(model_predict, background, feature_names=[f"Feature_{i}" for i in range(sample_data_np.shape[1])])
    
        # Compute SHAP values
        shap_values = explainer.shap_values(sample_data_np)
    
        return shap_values

