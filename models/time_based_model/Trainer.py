# Trainer.py

import torch
from tqdm import tqdm
import wandb
import os

class Trainer:
    def __init__(self, model, optimizer, scheduler, criterion, device, patience, evaluator, output_dir):
        """
        Initializes the Trainer.
        
        Args:
            model (torch.nn.Module): The neural network model.
            optimizer (torch.optim.Optimizer): Optimizer for training.
            scheduler (torch.optim.lr_scheduler): Learning rate scheduler.
            criterion (torch.nn.Module): Loss function.
            device (torch.device): Device to run the model on.
            patience (int): Number of epochs to wait for improvement before early stopping.
            evaluator (Evaluator): Evaluator instance for computing metrics.
            output_dir (str): Directory to save model checkpoints.
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.patience = patience
        self.evaluator = evaluator
        self.output_dir = output_dir
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.train_losses = []
        self.val_losses = []

    def train(self, num_epochs, train_loader, val_loader):
        """
        Trains the model using the provided data loaders.
        
        Args:
            num_epochs (int): Number of training epochs.
            train_loader (DataLoader): DataLoader for training data.
            val_loader (DataLoader): DataLoader for validation data.
        """
        for epoch in range(num_epochs):
            self.model.train()
            total_train_loss = 0.0
            for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                y_pred = self.model(x)
                loss = self.criterion(y_pred, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.scheduler.step()
                total_train_loss += loss.item() * x.size(0)

            avg_train_loss = total_train_loss / len(train_loader.dataset)
            self.train_losses.append(avg_train_loss)

            # Evaluate on validation set
            val_loss = self.evaluator.evaluate_metrics(val_loader)
            self.val_losses.append(val_loss)

            # Log metrics to WandB
            wandb.log({
                "train_loss": avg_train_loss,
                "val_loss": val_loss,
                "epoch": epoch + 1
            })

            print(f'Epoch [{epoch+1}/{num_epochs}] - Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}')

            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
                # Save the best model
                torch.save(self.model.state_dict(), os.path.join(self.output_dir, 'best_model.pth'))
                print("Model improved and saved.")
            else:
                self.epochs_without_improvement += 1
                print(f"No improvement for {self.epochs_without_improvement} epochs.")
                if self.epochs_without_improvement >= self.patience:
                    print(f'Early stopping after {self.patience} epochs without improvement.')
                    break
