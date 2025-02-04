"""
Visualize the test solar storm dataset
"""
from datasets import Dataset
import datasets
import matplotlib.pyplot as plt
import pandas as pd
import os
from tqdm import tqdm
import random
train_ds = Dataset.load_from_disk("../../data/akebono_solar_combined_v7_chu_train")
test_ds = Dataset.load_from_disk("../../data/akebono_solar_combined_v7_chu_test")
ds = datasets.concatenate_datasets([train_ds, test_ds])
# Order all ds chronologically
ds = ds.sort("DateTimeFormatted")

# Filter data
# ds = ds.filter(lambda x: pd.Timestamp(x["DateTimeFormatted"]).day == 6 and pd.Timestamp(x["DateTimeFormatted"]).month == 5, num_proc=os.cpu_count())
ds = ds.filter(lambda x: (pd.Timestamp(x["DateTimeFormatted"]).day >= 2 and 
                         pd.Timestamp(x["DateTimeFormatted"]).day <= 7 and
                         pd.Timestamp(x["DateTimeFormatted"]).month == 6 and
                         pd.Timestamp(x["DateTimeFormatted"]).year == 1991), 
               num_proc=os.cpu_count())
# ds = ds.filter(lambda x: pd.Timestamp(x["DateTimeFormatted"]).year == 1990, num_proc=os.cpu_count())


# Convert to pandas dataframe for easier plotting
df = ds.to_pandas()
df['DateTimeFormatted'] = pd.to_datetime(df['DateTimeFormatted'])

# Group by date
dates = df['DateTimeFormatted'].dt.date.unique()


# Create separate plots for each hour
for idx, date in tqdm(enumerate(dates)):
    # Filter data for this date
    mask = df['DateTimeFormatted'].dt.date == date
    day_data = df[mask]
    
    # Get unique hours for this day
    hours = day_data['DateTimeFormatted'].dt.hour.unique()

    # Create a new figure with subplots for the day
    fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12, ax13, ax14, ax15) = plt.subplots(15, 1, figsize=(14, 64))
    
    # Plot Te1
    ax1.plot(day_data['DateTimeFormatted'], day_data['Te1'], '.', label='Te1')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Electron Temperature (K)')
    ax1.set_title(f'Te1 vs Time - {date}')
    ax1.grid(True)
    ax1.tick_params(axis='x', rotation=45)
    ax1.legend()

    # Plot Te2
    ax2.plot(day_data['DateTimeFormatted'], day_data['Te2'], '.', label='Te2')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Electron Temperature (K)')
    ax2.set_title(f'Te2 vs Time - {date}')
    ax2.grid(True)
    ax2.tick_params(axis='x', rotation=45)
    ax2.legend()

    # Plot Te3
    ax3.plot(day_data['DateTimeFormatted'], day_data['Te3'], '.', label='Te3')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Electron Temperature (K)')
    ax3.set_title(f'Te3 vs Time - {date}')
    ax3.grid(True)
    ax3.tick_params(axis='x', rotation=45)
    ax3.legend()

    # Plot Ne1
    ax4.plot(day_data['DateTimeFormatted'], day_data['Ne1'], '.', label='Ne1')
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Electron Density (m⁻³)')
    ax4.set_title(f'Ne1 vs Time - {date}')
    ax4.grid(True)
    ax4.tick_params(axis='x', rotation=45)
    ax4.legend()

    # Plot Ne2
    ax5.plot(day_data['DateTimeFormatted'], day_data['Ne2'], '.', label='Ne2')
    ax5.set_xlabel('Time')
    ax5.set_ylabel('Electron Density (m⁻³)')
    ax5.set_title(f'Ne2 vs Time - {date}')
    ax5.grid(True)
    ax5.tick_params(axis='x', rotation=45)
    ax5.legend()

    # Plot Ne3
    ax6.plot(day_data['DateTimeFormatted'], day_data['Ne3'], '.', label='Ne3')
    ax6.set_xlabel('Time')
    ax6.set_ylabel('Electron Density (m⁻³)')
    ax6.set_title(f'Ne3 vs Time - {date}')
    ax6.grid(True)
    ax6.tick_params(axis='x', rotation=45)
    ax6.legend()
    
    # Plot indices on middle subplot
    # Plot AL index
    # Plot all AL indices
    for i in range(31):
        if i == 0:
            ax7.plot(day_data['DateTimeFormatted'], day_data[f'AL_index_{i}'], '.', color='blue', label='AL Index')
        else:
            ax7.plot(day_data['DateTimeFormatted'], day_data[f'AL_index_{i}'], '.', color='blue')
    # Plot all SYM-H indices
    for i in range(145):
        if i == 0:
            ax7.plot(day_data['DateTimeFormatted'], day_data[f'SYM_H_{i}'], '.', color='orange', label='SYM-H')
        else:
            ax7.plot(day_data['DateTimeFormatted'], day_data[f'SYM_H_{i}'], '.', color='orange')
    
    # Plot all F10.7 indices
    for i in range(4):
        if i == 0:
            ax7.plot(day_data['DateTimeFormatted'], day_data[f'f107_index_{i}'], '.', color='green', label='F10.7')
        else:
            ax7.plot(day_data['DateTimeFormatted'], day_data[f'f107_index_{i}'], '.', color='green')
    ax7.set_xlabel('Time')
    ax7.set_ylabel('Index Values')
    ax7.grid(True)
    ax7.tick_params(axis='x', rotation=45)
    ax7.legend()

    # Plot altitude on fourth subplot
    ax8.plot(day_data['DateTimeFormatted'], day_data['Altitude'], '.', label='Altitude')
    ax8.set_xlabel('Time')
    ax8.set_ylabel('Altitude (km)')
    ax8.grid(True)
    ax8.tick_params(axis='x', rotation=45)
    ax8.legend()

    # Plot GCLAT
    ax9.plot(day_data['DateTimeFormatted'], day_data['GCLAT'], '.', label='GCLAT')
    ax9.set_xlabel('Time')
    ax9.set_ylabel('GCLAT')
    ax9.grid(True)
    ax9.tick_params(axis='x', rotation=45)
    ax9.legend()

    # Plot GCLON
    ax10.plot(day_data['DateTimeFormatted'], day_data['GCLON'], '.', label='GCLON')
    ax10.set_xlabel('Time')
    ax10.set_ylabel('GCLON')
    ax10.grid(True)
    ax10.tick_params(axis='x', rotation=45)
    ax10.legend()

    # Plot ILAT
    ax11.plot(day_data['DateTimeFormatted'], day_data['ILAT'], '.', label='ILAT')
    ax11.set_xlabel('Time')
    ax11.set_ylabel('ILAT')
    ax11.grid(True)
    ax11.tick_params(axis='x', rotation=45)
    ax11.legend()

    # Plot GLAT
    ax12.plot(day_data['DateTimeFormatted'], day_data['GLAT'], '.', label='GLAT')
    ax12.set_xlabel('Time')
    ax12.set_ylabel('GLAT')
    ax12.grid(True)
    ax12.tick_params(axis='x', rotation=45)
    ax12.legend()

    # Plot GMLT
    ax13.plot(day_data['DateTimeFormatted'], day_data['GMLT'], '.', label='GMLT')
    ax13.set_xlabel('Time')
    ax13.set_ylabel('GMLT')
    ax13.grid(True)
    ax13.tick_params(axis='x', rotation=45)
    ax13.legend()

    # Plot XXLAT
    ax14.plot(day_data['DateTimeFormatted'], day_data['XXLAT'], '.', label='XXLAT')
    ax14.set_xlabel('Time')
    ax14.set_ylabel('XXLAT')
    ax14.grid(True)
    ax14.tick_params(axis='x', rotation=45)
    ax14.legend()

    # Plot XXLON
    ax15.plot(day_data['DateTimeFormatted'], day_data['XXLON'], '.', label='XXLON')
    ax15.set_xlabel('Time')
    ax15.set_ylabel('XXLON')
    ax15.grid(True)
    ax15.tick_params(axis='x', rotation=45)
    ax15.legend()
    plt.tight_layout()
    
    # Save the day's plot
    plt.savefig(f'plot/storm-{date}.png', dpi=300, bbox_inches='tight')
    plt.close()
