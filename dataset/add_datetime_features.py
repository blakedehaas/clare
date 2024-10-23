import pandas as pd
import numpy as np
train_df = pd.read_csv('../data/train_v4.tsv', sep='\t')
eval_df = pd.read_csv('../data/validation_v4.tsv', sep='\t')
test_df = pd.read_csv('../data/test_v4.tsv', sep='\t')

# Convert date/time into cyclic features for both train and eval datasets
for df in [test_df, train_df, eval_df]:
    # Convert DateTimeFormatted to datetime
    df['DateTimeFormatted'] = pd.to_datetime(df['DateTimeFormatted'])

    # Extract components from DateTimeFormatted
    df['Year_sin'] = np.sin(2 * np.pi * (df['DateTimeFormatted'].dt.year - df['DateTimeFormatted'].dt.year.min()) / 11)  # 11 year solar cycle
    df['DayOfYear'] = df['DateTimeFormatted'].dt.dayofyear
    df['TimeOfDay'] = (df['DateTimeFormatted'].dt.hour * 3600 +
                       df['DateTimeFormatted'].dt.minute * 60 +
                       df['DateTimeFormatted'].dt.second) / 86400

    # Calculate cyclic features
    df['DayOfYear_sin'] = np.sin(2 * np.pi * df['DayOfYear'] / 365.25)
    df['TimeOfDay_sin'] = np.sin(2 * np.pi * df['TimeOfDay'])

# Drop original date column and intermediate columns
test_df_final = test_df.drop(['DateTimeFormatted', 'DayOfYear', 'TimeOfDay'], axis=1)
train_df_final = train_df.drop(['DateTimeFormatted', 'DayOfYear', 'TimeOfDay'], axis=1)
eval_df_final = eval_df.drop(['DateTimeFormatted', 'DayOfYear', 'TimeOfDay'], axis=1)

test_df_final.to_csv('../data/test_v4_5.tsv', sep='\t', index=False)
train_df_final.to_csv('../data/train_v4_5.tsv', sep='\t', index=False)
eval_df_final.to_csv('../data/validation_v4_5.tsv', sep='\t', index=False)
