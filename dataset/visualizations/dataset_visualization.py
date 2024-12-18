"""
Visualize the test solar storm dataset
"""
from datasets import Dataset
import matplotlib.pyplot as plt
import pandas as pd

ds = Dataset.load_from_disk("../../data/akebono_solar_combined_v7_chu_test")

# Order all ds chronologically
ds = ds.sort("DateTimeFormatted")


# Convert to pandas dataframe for easier plotting
df = ds.to_pandas()
df['DateTimeFormatted'] = pd.to_datetime(df['DateTimeFormatted'])

# Group by date
dates = df['DateTimeFormatted'].dt.date.unique()
# Create separate plots for each day
for date in dates:
    # Filter data for this date
    mask = df['DateTimeFormatted'].dt.date == date
    day_data = df[mask]
    
    # Create a new figure with subplots for each day
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 16), height_ratios=[2, 1, 1])
    
    # Plot electron temperature on top subplot
    ax1.plot(day_data['DateTimeFormatted'], day_data['Te1'], '-')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Electron Temperature (K)')
    ax1.set_title(f'Electron Temperature vs Time - {date}')
    ax1.grid(True)
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot indices on middle subplot
    # Plot AL index
    ax2.plot(day_data['DateTimeFormatted'], day_data['AL_index_0'], '-', color='blue', label='AL Index')
    # Plot all SYM-H indices
    for i in range(145):  # 0-144 based on input_columns from context
        if i == 0:
            ax2.plot(day_data['DateTimeFormatted'], day_data[f'SYM_H_{i}'], '-', color='orange', label='SYM-H')
        else:
            ax2.plot(day_data['DateTimeFormatted'], day_data[f'SYM_H_{i}'], '-', color='orange')
    
    # Plot all F10.7 indices
    for i in range(4):  # 0-3 based on input_columns from context
        if i == 0:
            ax2.plot(day_data['DateTimeFormatted'], day_data[f'f107_index_{i}'], '-', color='green', label='F10.7')
        else:
            ax2.plot(day_data['DateTimeFormatted'], day_data[f'f107_index_{i}'], '-', color='green')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Index Values')
    ax2.grid(True)
    ax2.tick_params(axis='x', rotation=45)
    ax2.legend()
    
    # Plot location parameters on bottom subplot
    ax3.plot(day_data['DateTimeFormatted'], day_data['Altitude'], '-', label='Altitude')
    ax3.plot(day_data['DateTimeFormatted'], day_data['GCLAT'], '-', label='GCLAT')
    ax3.plot(day_data['DateTimeFormatted'], day_data['GCLON'], '-', label='GCLON')
    ax3.plot(day_data['DateTimeFormatted'], day_data['ILAT'], '-', label='ILAT')
    ax3.plot(day_data['DateTimeFormatted'], day_data['GLAT'], '-', label='GLAT')
    ax3.plot(day_data['DateTimeFormatted'], day_data['GMLT'], '-', label='GMLT')
    ax3.plot(day_data['DateTimeFormatted'], day_data['XXLAT'], '-', label='XXLAT')
    ax3.plot(day_data['DateTimeFormatted'], day_data['XXLON'], '-', label='XXLON')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Location Parameters')
    ax3.grid(True)
    ax3.tick_params(axis='x', rotation=45)
    ax3.legend()
    
    plt.tight_layout()
    
    # Save each day's plot
    plt.savefig(f'raw_data_solar_storm_{date}.png', dpi=300, bbox_inches='tight')
    plt.close()



