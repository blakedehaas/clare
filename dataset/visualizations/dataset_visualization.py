"""
Visualize the test solar storm dataset
"""
from datasets import Dataset
import matplotlib.pyplot as plt
import pandas as pd
import os

ds = Dataset.load_from_disk("../../data/akebono_solar_combined_v7_chu_train")

# Order all ds chronologically
ds = ds.sort("DateTimeFormatted")

# Filter for Jan 23 1997
start_date = pd.Timestamp('1995-01-23')
end_date = pd.Timestamp('1995-01-24') 
ds = ds.filter(lambda x: start_date <= pd.Timestamp(x["DateTimeFormatted"]) < end_date, num_proc=os.cpu_count())

# Convert to pandas dataframe for easier plotting
df = ds.to_pandas()
df['DateTimeFormatted'] = pd.to_datetime(df['DateTimeFormatted'])

# Group by date
dates = df['DateTimeFormatted'].dt.date.unique()
# Create separate plots for each hour
for date in dates:
    # Filter data for this date
    mask = df['DateTimeFormatted'].dt.date == date
    day_data = df[mask]
    
    # Get unique hours for this day
    hours = day_data['DateTimeFormatted'].dt.hour.unique()
    
    for hour in hours:
        # Filter data for this hour
        hour_mask = day_data['DateTimeFormatted'].dt.hour == hour
        hour_data = day_data[hour_mask]
        
        # Create a new figure with subplots for each hour
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 20))
        
        # Plot electron temperature on top subplot
        ax1.plot(hour_data['DateTimeFormatted'], hour_data['Te1'], '-')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Electron Temperature (K)')
        ax1.set_title(f'Electron Temperature vs Time - {date} Hour {hour:02d}:00')
        ax1.grid(True)
        ax1.tick_params(axis='x', rotation=45)
        

        # Plot indices on middle subplot
        # Plot AL index
        # Plot all AL indices
        for i in range(31):
            if i == 0:
                ax2.plot(hour_data['DateTimeFormatted'], hour_data[f'AL_index_{i}'], '-', color='blue', label='AL Index')
            else:
                ax2.plot(hour_data['DateTimeFormatted'], hour_data[f'AL_index_{i}'], '-', color='blue')
        # Plot all SYM-H indices
        for i in range(145):
            if i == 0:
                ax2.plot(hour_data['DateTimeFormatted'], hour_data[f'SYM_H_{i}'], '-', color='orange', label='SYM-H')
            else:
                ax2.plot(hour_data['DateTimeFormatted'], hour_data[f'SYM_H_{i}'], '-', color='orange')
        
        # Plot all F10.7 indices
        for i in range(4):
            if i == 0:
                ax2.plot(hour_data['DateTimeFormatted'], hour_data[f'f107_index_{i}'], '-', color='green', label='F10.7')
            else:
                ax2.plot(hour_data['DateTimeFormatted'], hour_data[f'f107_index_{i}'], '-', color='green')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Index Values')
        ax2.grid(True)
        ax2.tick_params(axis='x', rotation=45)
        ax2.legend()

        # Plot altitude on third subplot
        ax3.plot(hour_data['DateTimeFormatted'], hour_data['Altitude'], '-', label='Altitude')
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Altitude (km)')
        ax3.grid(True)
        ax3.tick_params(axis='x', rotation=45)
        ax3.legend()

        # Plot location parameters on fourth subplot
        ax4.plot(hour_data['DateTimeFormatted'], hour_data['GCLAT'], '-', label='GCLAT')
        ax4.plot(hour_data['DateTimeFormatted'], hour_data['GCLON'], '-', label='GCLON')
        ax4.plot(hour_data['DateTimeFormatted'], hour_data['ILAT'], '-', label='ILAT')
        ax4.plot(hour_data['DateTimeFormatted'], hour_data['GLAT'], '-', label='GLAT')
        ax4.plot(hour_data['DateTimeFormatted'], hour_data['GMLT'], '-', label='GMLT')
        ax4.plot(hour_data['DateTimeFormatted'], hour_data['XXLAT'], '-', label='XXLAT')
        ax4.plot(hour_data['DateTimeFormatted'], hour_data['XXLON'], '-', label='XXLON')
        ax4.set_xlabel('Time')
        ax4.set_ylabel('Location Parameters')
        ax4.grid(True)
        ax4.tick_params(axis='x', rotation=45)
        ax4.legend()
        plt.tight_layout()
        
        # Save each hour's plot
        plt.savefig(f'raw_data_{date}_hour_{hour:02d}.png', dpi=300, bbox_inches='tight')
        plt.close()
