# Visualization.py

import os
import matplotlib.pyplot as plt
import shap
import numpy as np
import scipy.stats as stats
import json

class Visualization:
    def __init__(self, output_dir):
        """
        Initializes the Visualization module.
        
        Args:
            output_dir (str): Directory where plots will be saved.
        """
        self.output_dir = output_dir

    def plot_training_validation_loss(self, train_losses, val_losses, output_name='loss_plot.png'):
        """
        Plots training and validation loss over epochs.
        
        Args:
            train_losses (list of float): List of training losses.
            val_losses (list of float): List of validation losses.
            output_name (str): Filename for the saved plot.
        """
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(train_losses) + 1)
        plt.plot(epochs, train_losses, 'bo-', label='Training Loss')
        plt.plot(epochs, val_losses, 'ro-', label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, output_name))
        plt.close()

    def plot_residuals_histogram(self, residuals, output_name='residuals_histogram.png'):
        """
        Plots a histogram of residuals with statistical annotations for performance, bias, homoscedasticity, and non-linearity.
        
        Args:
            residuals (np.ndarray): Array of residuals.
            output_name (str): Filename for the saved plot.
        """
        plt.figure(figsize=(12, 8))
        
        if residuals.ndim == 1:
            residuals = residuals.reshape(-1, 1)
        
        num_outputs = residuals.shape[1]
        colors = plt.cm.viridis(np.linspace(0, 1, num_outputs))
        
        metrics = {}
        
        for i in range(num_outputs):
            plt.hist(residuals[:, i], bins=50, alpha=0.6, color=colors[i], label=f'Output {i+1}')
            
            # Calculate statistics
            mean = np.mean(residuals[:, i])
            median = np.median(residuals[:, i])
            std = np.std(residuals[:, i])
            skewness = stats.skew(residuals[:, i])
            kurt = stats.kurtosis(residuals[:, i])
            mae = np.mean(np.abs(residuals[:, i]))
            mse = np.mean(residuals[:, i]**2)
            rmse = np.sqrt(mse)
            
            metrics[f'Output {i+1}'] = {
                'Mean': mean,
                'Median': median,
                'Std Dev': std,
                'Skewness': skewness,
                'Kurtosis': kurt,
                'MAE': mae,
                'MSE': mse,
                'RMSE': rmse
            }
            
            # Plot mean and median lines
            plt.axvline(mean, color=colors[i], linestyle='dashed', linewidth=1.5, label=f'Output {i+1} Mean')
            plt.axvline(median, color=colors[i], linestyle='solid', linewidth=1.5, label=f'Output {i+1} Median')
            
            # Annotate statistics
            plt.text(mean, plt.ylim()[1]*0.9, f'Mean: {mean:.2f}', color=colors[i], fontsize=9, ha='left')
            plt.text(median, plt.ylim()[1]*0.85, f'Median: {median:.2f}', color=colors[i], fontsize=9, ha='left')
            plt.text(mean, plt.ylim()[1]*0.8, f'Std: {std:.2f}', color=colors[i], fontsize=9, ha='left')
            plt.text(mean, plt.ylim()[1]*0.75, f'Skew: {skewness:.2f}', color=colors[i], fontsize=9, ha='left')
            plt.text(mean, plt.ylim()[1]*0.7, f'Kurt: {kurt:.2f}', color=colors[i], fontsize=9, ha='left')
            plt.text(mean, plt.ylim()[1]*0.65, f'MAE: {mae:.2f}', color=colors[i], fontsize=9, ha='left')
            plt.text(mean, plt.ylim()[1]*0.6, f'RMSE: {rmse:.2f}', color=colors[i], fontsize=9, ha='left')
        
        # Plot zero residual line
        plt.axvline(0, color='black', linestyle='dashdot', linewidth=1.5, label='Zero Residual')
        
        plt.title('Residuals Histogram with Statistical Annotations')
        plt.xlabel('Residual')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, output_name))
        plt.close()
        
        # Save metrics to a text file for further analysis
        metrics_path = os.path.join(self.output_dir, 'residuals_metrics.json')
        try:
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=4)
            print(f"[INFO] Residuals metrics saved to {metrics_path}")
        except Exception as e:
            print(f"[ERROR] Failed to save residuals metrics: {e}")

    def plot_residuals_vs_predicted(self, residuals, predictions, output_name='residuals_vs_predicted.png'):
        """
        Plots residuals versus predicted values to assess homoscedasticity and non-linearity.
        
        Args:
            residuals (np.ndarray): Array of residuals.
            predictions (np.ndarray): Array of predicted values.
            output_name (str): Filename for the saved plot.
        """
        plt.figure(figsize=(10, 6))
        plt.scatter(predictions, residuals, alpha=0.5, color='blue', edgecolors='k')
        plt.axhline(0, color='red', linestyle='dashed', linewidth=2)
        plt.title('Residuals vs. Predicted Values')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, output_name))
        plt.close()

    def plot_residuals_qq(self, residuals, output_name='residuals_qq_plot.png'):
        """
        Plots a QQ plot of residuals to assess normality.
        
        Args:
            residuals (np.ndarray): Array of residuals.
            output_name (str): Filename for the saved plot.
        """
        plt.figure(figsize=(10, 6))
        stats.probplot(residuals.flatten(), dist="norm", plot=plt)
        plt.title('QQ Plot of Residuals')
        plt.xlabel('Theoretical Quantiles')
        plt.ylabel('Ordered Residuals')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, output_name))
        plt.close()

    def plot_shap_summary(self, shap_values, feature_names, output_name='shap_summary.png'):
        """
        Plots SHAP summary plot.
        
        Args:
            shap_values (shap.Explanation or list of shap.Explanation): SHAP values.
            feature_names (list of str): List of feature names.
            output_name (str): Filename for the saved plot.
        """
        plt.figure()
        shap.summary_plot(shap_values, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, output_name))
        plt.close()

    def generate_plots(self, residuals=None, predictions=None, shap_values=None, feature_names=None, 
                      train_losses=None, val_losses=None, plot_types=None):
        """
        Interface method to generate specified plots or all plots by default.
        
        Args:
            residuals (np.ndarray, optional): Array of residuals for residual plots.
            predictions (np.ndarray, optional): Array of predicted values for residuals vs predicted plot.
            shap_values (shap.Explanation or list of shap.Explanation, optional): SHAP values for SHAP summary plot.
            feature_names (list of str, optional): List of feature names for SHAP summary plot.
            train_losses (list of float, optional): List of training losses for loss plot.
            val_losses (list of float, optional): List of validation losses for loss plot.
            plot_types (list of str, optional): List of plot types to generate. 
                Supported plot types:
                    - 'training_validation_loss'
                    - 'residuals_histogram'
                    - 'residuals_vs_predicted'
                    - 'residuals_qq_plot'
                    - 'shap_summary'
        """
        supported_plots = {
            'training_validation_loss': {
                'method': self.plot_training_validation_loss,
                'args': {
                    'train_losses': train_losses,
                    'val_losses': val_losses
                }
            },
            'residuals_histogram': {
                'method': self.plot_residuals_histogram,
                'args': {
                    'residuals': residuals
                }
            },
            'residuals_vs_predicted': {
                'method': self.plot_residuals_vs_predicted,
                'args': {
                    'residuals': residuals,
                    'predictions': predictions
                }
            },
            'residuals_qq_plot': {
                'method': self.plot_residuals_qq,
                'args': {
                    'residuals': residuals
                }
            },
            'shap_summary': {
                'method': self.plot_shap_summary,
                'args': {
                    'shap_values': shap_values,
                    'feature_names': feature_names
                }
            }
        }

        if plot_types is None:
            plot_types = list(supported_plots.keys())

        for plot_type in plot_types:
            if plot_type not in supported_plots:
                print(f"[WARNING] Plot type '{plot_type}' is not supported. Skipping.")
                continue
            
            plot_info = supported_plots[plot_type]
            method = plot_info['method']
            args = plot_info['args']
            
            # Check if all required arguments are provided
            missing_args = [arg for arg, value in args.items() if value is None]
            if missing_args:
                print(f"[WARNING] Missing arguments {missing_args} for plot '{plot_type}'. Skipping.")
                continue
            
            try:
                method(**args)
                print(f"[INFO] '{plot_type}' plot generated and saved.")
            except Exception as e:
                print(f"[ERROR] Failed to generate '{plot_type}' plot: {e}")
