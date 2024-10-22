import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import wandb  # Ensure wandb is imported

def evaluate_model(model, data_loader, criterion, device, target_mean=None, target_std=None, fold=None):
    model.eval()
    total_loss = 0
    predictions = []
    true_values = []

    with torch.no_grad():
        for x, y in tqdm(data_loader, desc="Evaluating"):
            x = x.to(device)
            y = y.to(device)

            # Forward pass
            y_pred = model(x)
            loss = criterion(y_pred, y)
            total_loss += loss.item()

            # Collect predictions and true values
            predictions.extend(y_pred.cpu().numpy())
            true_values.extend(y.cpu().numpy())

    average_loss = total_loss / len(data_loader)

    # Convert to numpy arrays
    predictions = np.array(predictions)
    true_values = np.array(true_values)

    # Unnormalize if necessary
    if target_mean is not None and target_std is not None:
        predictions = predictions * target_std.values + target_mean.values
        true_values = true_values * target_std.values + target_mean.values

    # Flatten the arrays for metric calculations
    predictions_flat = predictions.flatten()
    true_values_flat = true_values.flatten()

    # Compute metrics
    mse = mean_squared_error(true_values_flat, predictions_flat)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true_values_flat, predictions_flat)
    r2 = r2_score(true_values_flat, predictions_flat)
    mape = np.mean(np.abs((true_values_flat - predictions_flat) / (true_values_flat + 1e-8))) * 100  # Avoid division by zero

    # Print metrics
    print(f"Fold {fold}:")
    print(f"Average Loss: {average_loss:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R² Score: {r2:.4f}")
    print(f"MAPE: {mape:.2f}%")

    # Log metrics to wandb
    if wandb.run is not None:
        wandb.log({
            "average_loss": average_loss,
            "MSE": mse,
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2,
            "MAPE": mape
        })

    # Plot predictions vs. true values
    plt.figure(figsize=(8, 8))
    plt.scatter(true_values_flat, predictions_flat, alpha=0.5)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title(f'Predictions vs. True Values (Fold {fold})')
    plt.plot([true_values_flat.min(), true_values_flat.max()], [true_values_flat.min(), true_values_flat.max()], 'r--')
    plt.tight_layout()
    plt.savefig(f'pred_vs_true_fold_{fold}.png')
    plt.close()

    return average_loss
