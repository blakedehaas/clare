import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.dates as mdates
from scipy import stats


class Visualizer:
    def __init__(self, csv_path, model_name="1_24", output_dir="."):
        """
        Visualizer class that loads evaluation results from CSV
        and generates plots.
        """
        self.csv_path = csv_path
        self.model_name = model_name
        self.output_dir = output_dir
        self.results_df = pd.read_csv(csv_path)

        # Extract columns
        self.predictions = self.results_df["predictions"].to_numpy()
        self.true_values = self.results_df["true_values"].to_numpy()
        self.deviations = self.results_df["deviations"].to_numpy()

        # Parse DateTimeFormatted column for the time-series plot
        if "DateTimeFormatted" in self.results_df.columns:
            self.dates = pd.to_datetime(self.results_df["DateTimeFormatted"])
        else:
            raise ValueError("The column 'DateTimeFormatted' is missing from the dataset.")

    def plot_deviations_histogram(self):
        """
        Plots a histogram of the deviations.
        """
        plt.figure(figsize=(12, 8))
        plt.hist(self.deviations, bins=50, edgecolor='black')
        plt.xlabel('Deviation from Ground Truth')
        plt.ylabel('Frequency')
        plt.title('Distribution of Model Predictions Deviation')

        # Add Deviation Percentages
        thresholds = [100, 200, 300, 500, 1000, 2000, 5000]
        percentages = [
            sum(abs(dev) <= t for dev in self.deviations) / len(self.deviations) * 100
            for t in thresholds
        ]
        text = "\n".join([f"Within {t}: {p:.2f}%" for t, p in zip(thresholds, percentages)])
        plt.text(
            0.95,
            0.95,
            text,
            transform=plt.gca().transAxes,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
        )
        plt.tight_layout()

        out_file = os.path.join(self.output_dir, f"{self.model_name}_deviation.png")
        plt.savefig(out_file)
        plt.close()

    def plot_correlation(self, max_val=15000):
        """
        Plots a 2D histogram of predictions vs. true values,
        shows diagonal, RMSE, and correlation.
        """
        plt.figure(figsize=(10, 8))
        h = plt.hist2d(
            self.predictions,
            self.true_values,
            bins=100,
            norm=LogNorm(),
            cmap="viridis",
            range=[[0, max_val], [0, max_val]],
        )
        plt.colorbar(h[3], label="Obs#")
        plt.xlabel("Te$_{model}$ [K]")
        plt.ylabel("Te$_{obs}$ [K]")
        plt.title("Observed vs Predicted Electron Temperature")
        plt.plot([0, max_val], [0, max_val], "r--", alpha=0.75)

        rmse = np.sqrt(np.mean((self.predictions - self.true_values) ** 2))
        r = np.corrcoef(self.predictions, self.true_values)[0, 1]
        plt.text(
            0.05,
            0.95,
            f"RMSE={rmse:.3f}\nr={r:.4f}",
            transform=plt.gca().transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )
        plt.tight_layout()

        out_file = os.path.join(self.output_dir, f"{self.model_name}_correlation.png")
        plt.savefig(out_file, dpi=300)
        plt.close()

    def plot_time_series(self):
        """
        Plots Actual vs Predicted values over time and shows residuals.
        """
        fig, ax1 = plt.subplots(figsize=(14, 7))

        # Plot Actual and Predicted values using dates as the x-axis
        ax1.plot(self.dates, self.true_values, label="Actual", color="blue", linewidth=1)
        ax1.plot(
            self.dates,
            self.predictions,
            label="Predicted",
            color="red",
            linestyle="--",
            linewidth=1,
        )
        ax1.set_xlabel("Date & Time")
        ax1.set_ylabel("Electron Temperature (K)", color="black")
        ax1.tick_params(axis="y", labelcolor="black")
        ax1.set_title("Actual vs Predicted Electron Temperature Over Time")
        ax1.legend(loc="upper left")

        # Plot Residuals on a secondary y-axis
        ax2 = ax1.twinx()
        ax2.plot(
            self.dates,
            self.deviations,
            label="Residual",
            color="green",
            linestyle="-",
            linewidth=0.8,
        )
        ax2.set_ylabel("Residuals", color="green")
        ax2.tick_params(axis="y", labelcolor="green")
        ax2.axhline(0, color="black", linestyle="--", linewidth=0.8)

        # Format the x-axis for dates
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
        ax1.xaxis.set_major_locator(mdates.HourLocator(interval=12))
        fig.autofmt_xdate()  # Rotate and format x-axis labels

        fig.tight_layout()
        plt.grid(visible=True, linestyle="--", linewidth=0.5)

        out_file = os.path.join(self.output_dir, f"{self.model_name}_time_series_with_dates.png")
        plt.savefig(out_file, dpi=300)
        plt.show()

    def plot_deviation_vs_truth(self):
        """
        Plots a 2D histogram of the ground truth vs. the model deviation.
        """
        plt.figure(figsize=(10, 8))
        h = plt.hist2d(
            self.true_values,
            self.deviations,
            bins=100,
            norm=LogNorm(),
            cmap="viridis",
        )
        plt.colorbar(h[3], label="Obs#")
        plt.xlabel("Te$_{obs}$ [K]")
        plt.ylabel("Te$_{model}$ - Te$_{obs}$ [K]")
        plt.title("Deviation vs Ground Truth")

        # Compute and plot the mean deviation in bins of true_values
        bin_means, bin_edges, _ = stats.binned_statistic(
            self.true_values, self.deviations, statistic="mean", bins=50
        )
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        plt.plot(bin_centers, bin_means, "r-", linewidth=2, label="Mean Deviation")
        plt.legend()

        mean_deviation = np.mean(self.deviations)
        print(f"Mean Deviation: {mean_deviation:.3f}")
        plt.tight_layout()

        out_file = os.path.join(self.output_dir, f"{self.model_name}_deviation_vs_truth.png")
        plt.savefig(out_file, dpi=300)
        plt.close()


def main():
    # Adjust the path below to the CSV file generated by evaluate.py
    model_name = "1_24"
    output_dir = "./output"  # same common folder
    csv_path = os.path.join(output_dir, f"evaluation_results_{model_name}.csv")

    # Make sure output_dir exists
    os.makedirs(output_dir, exist_ok=True)

    visualizer = Visualizer(csv_path, model_name=model_name, output_dir=output_dir)

    # Generate plots
    visualizer.plot_deviations_histogram()
    visualizer.plot_correlation(max_val=15000)
    visualizer.plot_time_series()
    visualizer.plot_deviation_vs_truth()


if __name__ == "__main__":
    main()
