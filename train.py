import pandas as pd
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
import wandb

# Hyperparameters
batch_size = 1024
num_epochs = 1
max_lr = 3e-4
min_lr = 1e-5
model_name = 'tf_1'
eval_every = 1
log_every_step = 100

input_columns = ['Altitude', 'GCLAT', 'GCLON', 'ILAT', 'GLAT', 'GMLT', 'XXLAT', 'XXLON', 'Ne1', 'Pv1', 'I1', 'Year', 'DayOfYear_sin', 'TimeOfDay_sin']
output_column = 'Te1'
train_df = pd.read_csv('data/train.tsv', sep='\t')
eval_df = pd.read_csv('data/validation.tsv', sep='\t')

# Keep only specific columns
columns_to_keep = ['Altitude', 'GCLAT', 'GCLON', 'ILAT', 'GLAT', 'GMLT', 'XXLAT', 'XXLON', 'Te1', 'Ne1', 'Pv1', 'I1', 'DateFormatted', 'TimeFormatted']
train_df = train_df[columns_to_keep]
eval_df = eval_df[columns_to_keep]

# Convert DateFormatted and TimeFormatted columns to datetime
train_df['DateFormatted'] = pd.to_datetime(train_df['DateFormatted'] + ' ' + train_df['TimeFormatted'], format='%Y-%m-%d %H:%M:%S')
eval_df['DateFormatted'] = pd.to_datetime(eval_df['DateFormatted'] + ' ' + eval_df['TimeFormatted'], format='%Y-%m-%d %H:%M:%S')
train_df.drop('TimeFormatted', axis=1, inplace=True)
eval_df.drop('TimeFormatted', axis=1, inplace=True)

# Normalize all location and atmospheric parameters (mean = 0 and std dev = 1)
columns_to_normalize = ['Altitude', 'GCLAT', 'GCLON', 'ILAT', 'GLAT', 'GMLT', 'XXLAT', 'XXLON', 'Ne1', 'Pv1', 'I1', 'Te1']

# Function to calculate mean and std dev for specified columns
def calculate_stats(df, columns):
    means = df[columns].mean()
    stds = df[columns].std()
    return means, stds

# Function to normalize specified columns in the DataFrame
def normalize_df(df, means, stds, columns):
    df[columns] = (df[columns] - means) / stds
    return df

# Calculate mean and std for the specified columns
means, stds = calculate_stats(train_df, columns_to_normalize)
train_df_norm = normalize_df(train_df, means, stds, columns_to_normalize)

means, stds = calculate_stats(eval_df, columns_to_normalize)
eval_df_norm = normalize_df(eval_df, means, stds, columns_to_normalize)

# Verify there are no NaNs in the normalized DataFrame
assert train_df_norm[columns_to_normalize].isna().sum().sum() == 0, "NaN values found in normalized data"
assert eval_df_norm[columns_to_normalize].isna().sum().sum() == 0, "NaN values found in normalized data"

# Convert date/time into cyclic features for both train and eval datasets
for df in [train_df_norm, eval_df_norm]:
    # Extract components from DateFormatted
    df['Year'] = df['DateFormatted'].dt.year - 1989
    df['DayOfYear'] = df['DateFormatted'].dt.dayofyear
    df['TimeOfDay'] = (df['DateFormatted'].dt.hour * 3600 +
                       df['DateFormatted'].dt.minute * 60 +
                       df['DateFormatted'].dt.second) / 86400

    # Calculate cyclic features
    df['DayOfYear_sin'] = np.sin(2 * np.pi * df['DayOfYear'] / 365.25)
    df['TimeOfDay_sin'] = np.sin(2 * np.pi * df['TimeOfDay'])

# Drop original date column and intermediate columns
train_df_norm_final = train_df_norm.drop(['DateFormatted', 'DayOfYear', 'TimeOfDay'], axis=1)
eval_df_norm_final = eval_df_norm.drop(['DateFormatted', 'DayOfYear', 'TimeOfDay'], axis=1)

class DataFrameDataset(Dataset):
    def __init__(self, dataframe, input_columns, output_column):
        self.X = torch.tensor(dataframe[input_columns].values, dtype=torch.float32)
        self.y = torch.tensor(dataframe[output_column].values, dtype=torch.float32).reshape(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class Network(nn.Module): # ff_1
    def __init__(self, input_size, hidden_size, output_size):
        super(Network, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.layer2(x)
        return x

class FF_2Network(nn.Module): # ff_2
    def __init__(self, input_size, hidden_size, output_size):
        super(FF_2Network, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size * 2)
        self.layer_norm2 = nn.LayerNorm(hidden_size * 2)
        self.layer3 = nn.Linear(hidden_size * 2, hidden_size * 4)
        self.layer_norm3 = nn.LayerNorm(hidden_size * 4)
        self.layer4 = nn.Linear(hidden_size * 4, hidden_size * 2)
        self.layer_norm4 = nn.LayerNorm(hidden_size * 2)
        self.layer5 = nn.Linear(hidden_size * 2, hidden_size)
        self.layer_norm5 = nn.LayerNorm(hidden_size)
        self.layer6 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()

    def forward(self, x):
        # First block
        x = self.layer1(x)
        x = self.layer_norm1(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Second block
        x = self.layer2(x)
        x = self.layer_norm2(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Third block
        x = self.layer3(x)
        x = self.layer_norm3(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Fourth block
        x = self.layer4(x)
        x = self.layer_norm4(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Fifth block
        x = self.layer5(x)
        x = self.layer_norm5(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Output layer
        x = self.layer6(x)
        return x

class TransformerRegressor(nn.Module): # tf_1
    def __init__(self, input_size, hidden_size=256, num_layers=4, num_heads=8, output_size=1, dropout=0.1):
        super(TransformerRegressor, self).__init__()
        self.linear = nn.Linear(1, hidden_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc1 = nn.Linear(hidden_size * input_size, hidden_size * 4)
        self.fc2 = nn.Linear(hidden_size * 4, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.layer_norm1 = nn.LayerNorm(hidden_size * 4)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.linear(x)  # Add sequence dimension
        x = self.dropout(x)
        x = self.transformer_encoder(x)
        x = x.reshape(x.size(0), -1)  # Flatten the output
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class GaussianNetwork(nn.Module): # gaussian_1
    def __init__(self, input_size, hidden_size, output_size):
        super(GaussianNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.mean_layer = nn.Linear(hidden_size, output_size)
        self.var_layer = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()


    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        mean = self.mean_layer(x)
        var = self.relu(self.var_layer(x))
        return mean, var

# Set up data loader
train_ds = DataFrameDataset(train_df_norm_final, input_columns, output_column)
eval_ds = DataFrameDataset(eval_df_norm_final, input_columns, output_column)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=8)
eval_loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False, num_workers=8)

# Initialize the model
input_size = len(input_columns)
hidden_size = 2048
output_size = 1
# model = Network(input_size, hidden_size, output_size).to("cuda")
# model = FF_2Network(input_size, hidden_size, output_size).to("cuda")
# model = GaussianNetwork(input_size, hidden_size, output_size).to("cuda")
model = TransformerRegressor(input_size).to("cuda")

# Define loss function and optimizer
criterion = nn.MSELoss()
# criterion = nn.GaussianNLLLoss()
optimizer = optim.Adam(model.parameters(), lr=max_lr)

# Implement One Cycle LR
steps_per_epoch = len(train_loader)
total_train_steps = num_epochs * steps_per_epoch
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, 
                                                total_steps=total_train_steps,
                                                pct_start=0.1, anneal_strategy='cos',
                                                cycle_momentum=True, base_momentum=0.85,
                                                max_momentum=0.95, div_factor=25.0,
                                                final_div_factor=10000.0)

# Initialize wandb run
wandb.init(
    project="auroral-precipitation-ml",
    config={
        "dataset_size": len(train_df),
        "validation_size": len(eval_df),
        "columns": columns_to_keep
    }
)

def evaluate_model(model, data_loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y in tqdm(data_loader):
            x = x.to("cuda")
            y = y.to("cuda")
            if model.__class__.__name__ == 'GaussianNetwork':
                y_pred, var = model(x)
                loss = criterion(y_pred, y, var)
            else:
                y_pred = model(x)
                loss = criterion(y_pred, y)

            total_loss += loss.item()
    return total_loss / len(data_loader)

# Training loop
total_steps = 0
for epoch in range(num_epochs):
    epoch_loss = 0
    model.train()
    for i, (x, y) in enumerate(tqdm(train_loader)):
        x = x.to("cuda")
        y = y.to("cuda")

        # Forward pass
        if model.__class__.__name__ == 'GaussianNetwork':
            y_pred, var = model(x)
            loss = criterion(y_pred, y, var)
        else:
            y_pred = model(x)
            loss = criterion(y_pred, y)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Step the scheduler
        scheduler.step()

        epoch_loss += loss.item()
        total_steps += 1

        # Log train loss and learning rate every log_every_step iterations
        if total_steps % log_every_step == 0:
            wandb.log({
                "train_loss": loss.item(),
                "learning_rate": scheduler.get_last_lr()[0],
                "total_steps": total_steps
            })

    avg_train_loss = epoch_loss / len(train_loader)
    test_loss = evaluate_model(model, eval_loader, criterion)
    
    # Log train and test loss to wandb
    wandb.log({
        "epoch": epoch,
        "test_loss": test_loss
    })
    if (epoch + 1) % eval_every == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Avg Train Loss: {avg_train_loss:.4f}, Test Loss: {test_loss:.4f}')

print('Training finished!')

# Save the model
torch.save(model.state_dict(), f'{model_name}.pth')

# Final evaluation
final_test_loss = evaluate_model(model, eval_loader, criterion)
print(f'Final Test Loss: {final_test_loss:.4f}')
wandb.log({"final_test_loss": final_test_loss}) # 0.0314

# Evaluate on the test set
# # Gaussian Network gaussian_1
# input_size = len(input_columns)
# hidden_size = 256
# model = GaussianNetwork(input_size, hidden_size, output_size).to("cuda")
# model.load_state_dict(torch.load('checkpoints/gaussian_1/gaussian_1.pth'))

# FF Network ff_1
# input_size = len(input_columns)
# hidden_size = 256
# model = Network(input_size, hidden_size, output_size).to("cuda")
# model.load_state_dict(torch.load('checkpoints/ff_1/ff_1.pth'))

# FF Network ff_2
# model = FF_2Network(input_size, hidden_size, output_size).to("cuda")
# model.load_state_dict(torch.load('checkpoints/ff_2/ff_2.pth'))

# Transformer Regressor tf_1
model = TransformerRegressor(input_size).to("cuda")
model.load_state_dict(torch.load('checkpoints/tf_1/tf_1.pth'))

model.eval()  # Set the model to evaluation mode
test_df = pd.read_csv('data/test.tsv', sep='\t')
test_df = test_df[columns_to_keep]
test_df['DateFormatted'] = pd.to_datetime(test_df['DateFormatted'] + ' ' + test_df['TimeFormatted'], format='%Y-%m-%d %H:%M:%S')
test_df.drop('TimeFormatted', axis=1, inplace=True)
test_df_norm = normalize_df(test_df, means, stds, columns_to_normalize)

# Convert date/time into cyclic features for test dataset
# Extract components from DateFormatted
test_df_norm['Year'] = test_df_norm['DateFormatted'].dt.year - 1989
test_df_norm['DayOfYear'] = test_df_norm['DateFormatted'].dt.dayofyear
test_df_norm['TimeOfDay'] = (test_df_norm['DateFormatted'].dt.hour * 3600 +
                             test_df_norm['DateFormatted'].dt.minute * 60 +
                             test_df_norm['DateFormatted'].dt.second) / 86400

# Calculate cyclic features
test_df_norm['DayOfYear_sin'] = np.sin(2 * np.pi * test_df_norm['DayOfYear'] / 365.25)
test_df_norm['TimeOfDay_sin'] = np.sin(2 * np.pi * test_df_norm['TimeOfDay'])
test_df_norm_final = test_df_norm.drop(['DateFormatted', 'DayOfYear', 'TimeOfDay'], axis=1)

test_ds = DataFrameDataset(test_df_norm_final, input_columns, output_column)

test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=8)

target_mean = means[output_column]
target_std = stds[output_column]

def unnormalize_mean(pred, target_mean, target_std):
    return pred * target_std + target_mean

def unnormalize_var(var, target_std):
    return var * (target_std ** 2)

correct_predictions = 0
total_samples = len(test_loader)
predictions = []
true_values = []

with torch.no_grad():
    for x, y in tqdm(test_loader, desc="Evaluating"):
        x = x.to("cuda")
        y = y.to("cuda")
        
        # Forward pass
        if model.__class__.__name__ == 'GaussianNetwork':
            y_pred, var = model(x)
        else:
            y_pred = model(x)
        
        # Unnormalize predictions and true values
        y_true = unnormalize_mean(y.cpu().item(), target_mean, target_std)
        y_pred = unnormalize_mean(y_pred.cpu().item(), target_mean, target_std)
        # var_pred = unnormalize_var(var.cpu().item(), target_std) # Not used but could be useful for uncertainty quantification
        
        predictions.append(y_pred)
        true_values.append(y_true)

# Calculate deviations
deviations = [pred - true for pred, true in zip(predictions, true_values)]

# Calculate percentages within specified absolute deviations
thresholds = [100, 200, 300, 500, 1000, 2000, 5000]
percentages = [
    sum(abs(dev) <= threshold for dev in deviations) / len(deviations) * 100
    for threshold in thresholds
]

# Calculate percentages within specified relative deviations
relative_thresholds = [5, 10, 15, 20]
relative_percentages = [
    sum(abs(dev) / true * 100 <= threshold for dev, true in zip(deviations, true_values)) / len(deviations) * 100
    for threshold in relative_thresholds
]

# Plot histogram
plt.figure(figsize=(12, 8))
plt.hist(deviations, bins=50, edgecolor='black')
plt.xlabel('Deviation from Ground Truth')
plt.ylabel('Frequency')
plt.title('Distribution of Model Predictions Deviation')

# Add text box with percentages
text = "\n".join([
    f"Within {threshold}: {percentage:.2f}%"
    for threshold, percentage in zip(thresholds, percentages)
] + ["\n"] + [  # Add an empty line between absolute and relative thresholds
    f"Within {threshold}%: {percentage:.2f}%"
    for threshold, percentage in zip(relative_thresholds, relative_percentages)
])
plt.text(0.95, 0.95, text, transform=plt.gca().transAxes, 
         verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()

# Save the plot
plt.savefig('deviation.png')
plt.close()  # Close the figure to free up memory