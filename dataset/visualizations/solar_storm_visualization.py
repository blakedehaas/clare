import datasets
import os
from tqdm import tqdm
import pandas as pd
import json
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import sys
sys.path.append("../..")
import utils
from models.feed_forward import FF_2Network


model_name = '1_24'
input_columns = ['Altitude', 'GCLAT', 'GCLON', 'ILAT', 'GLAT', 'GMLT', 'XXLAT', 'XXLON', 'AL_index_0', 'AL_index_1', 'AL_index_2', 'AL_index_3', 'AL_index_4', 'AL_index_5', 'AL_index_6', 'AL_index_7', 'AL_index_8', 'AL_index_9', 'AL_index_10', 'AL_index_11', 'AL_index_12', 'AL_index_13', 'AL_index_14', 'AL_index_15', 'AL_index_16', 'AL_index_17', 'AL_index_18', 'AL_index_19', 'AL_index_20', 'AL_index_21', 'AL_index_22', 'AL_index_23', 'AL_index_24', 'AL_index_25', 'AL_index_26', 'AL_index_27', 'AL_index_28', 'AL_index_29', 'AL_index_30', 'SYM_H_0', 'SYM_H_1', 'SYM_H_2', 'SYM_H_3', 'SYM_H_4', 'SYM_H_5', 'SYM_H_6', 'SYM_H_7', 'SYM_H_8', 'SYM_H_9', 'SYM_H_10', 'SYM_H_11', 'SYM_H_12', 'SYM_H_13', 'SYM_H_14', 'SYM_H_15', 'SYM_H_16', 'SYM_H_17', 'SYM_H_18', 'SYM_H_19', 'SYM_H_20', 'SYM_H_21', 'SYM_H_22', 'SYM_H_23', 'SYM_H_24', 'SYM_H_25', 'SYM_H_26', 'SYM_H_27', 'SYM_H_28', 'SYM_H_29', 'SYM_H_30', 'SYM_H_31', 'SYM_H_32', 'SYM_H_33', 'SYM_H_34', 'SYM_H_35', 'SYM_H_36', 'SYM_H_37', 'SYM_H_38', 'SYM_H_39', 'SYM_H_40', 'SYM_H_41', 'SYM_H_42', 'SYM_H_43', 'SYM_H_44', 'SYM_H_45', 'SYM_H_46', 'SYM_H_47', 'SYM_H_48', 'SYM_H_49', 'SYM_H_50', 'SYM_H_51', 'SYM_H_52', 'SYM_H_53', 'SYM_H_54', 'SYM_H_55', 'SYM_H_56', 'SYM_H_57', 'SYM_H_58', 'SYM_H_59', 'SYM_H_60', 'SYM_H_61', 'SYM_H_62', 'SYM_H_63', 'SYM_H_64', 'SYM_H_65', 'SYM_H_66', 'SYM_H_67', 'SYM_H_68', 'SYM_H_69', 'SYM_H_70', 'SYM_H_71', 'SYM_H_72', 'SYM_H_73', 'SYM_H_74', 'SYM_H_75', 'SYM_H_76', 'SYM_H_77', 'SYM_H_78', 'SYM_H_79', 'SYM_H_80', 'SYM_H_81', 'SYM_H_82', 'SYM_H_83', 'SYM_H_84', 'SYM_H_85', 'SYM_H_86', 'SYM_H_87', 'SYM_H_88', 'SYM_H_89', 'SYM_H_90', 'SYM_H_91', 'SYM_H_92', 'SYM_H_93', 'SYM_H_94', 'SYM_H_95', 'SYM_H_96', 'SYM_H_97', 'SYM_H_98', 'SYM_H_99', 'SYM_H_100', 'SYM_H_101', 'SYM_H_102', 'SYM_H_103', 'SYM_H_104', 'SYM_H_105', 'SYM_H_106', 'SYM_H_107', 'SYM_H_108', 'SYM_H_109', 'SYM_H_110', 'SYM_H_111', 'SYM_H_112', 'SYM_H_113', 'SYM_H_114', 'SYM_H_115', 'SYM_H_116', 'SYM_H_117', 'SYM_H_118', 'SYM_H_119', 'SYM_H_120', 'SYM_H_121', 'SYM_H_122', 'SYM_H_123', 'SYM_H_124', 'SYM_H_125', 'SYM_H_126', 'SYM_H_127', 'SYM_H_128', 'SYM_H_129', 'SYM_H_130', 'SYM_H_131', 'SYM_H_132', 'SYM_H_133', 'SYM_H_134', 'SYM_H_135', 'SYM_H_136', 'SYM_H_137', 'SYM_H_138', 'SYM_H_139', 'SYM_H_140', 'SYM_H_141', 'SYM_H_142', 'SYM_H_143', 'SYM_H_144', 'f107_index_0', 'f107_index_1', 'f107_index_2', 'f107_index_3']
output_columns = ['Te1']
test_ds = datasets.load_from_disk("/home/michael/auroral-precipitation-ml/data/akebono_solar_combined_v6_chu_test")
val_ds = datasets.load_from_disk("/home/michael/auroral-precipitation-ml/data/akebono_solar_combined_v6_chu_val")
train_subdirs = [f.path for f in os.scandir("/home/michael/auroral-precipitation-ml/data/akebono_solar_combined_v6_chu_train") if f.is_dir()]
train_ds = []
for subdir in tqdm(train_subdirs):
    train_ds.append(datasets.load_from_disk(subdir))
ds = datasets.concatenate_datasets([test_ds, val_ds, *train_ds])
ds = ds.remove_columns(['Ne1', 'Pv1', 'Te2', 'Ne2', 'Pv2', 'Te3', 'Ne3', 'Pv3', 'I1', 'I2', 'I3'])

# Model
input_size = len(input_columns)
hidden_size = 2048
output_size = 80
model = FF_2Network(input_size, hidden_size, output_size).to("cuda")
model.load_state_dict(torch.load(f'../../checkpoints/{model_name}.pth'))
model.eval()

# Delete all rows where DateTimeFormatted is not within June 2 1991 to June 7 1991 and order by time
start_date = pd.Timestamp('1991-06-02')
end_date = pd.Timestamp('1991-06-07')
ds = ds.filter(lambda x: start_date <= pd.Timestamp(x["DateTimeFormatted"]) <= end_date, num_proc=os.cpu_count())
ds = ds.sort("DateTimeFormatted")

# Normalize the dataset
with open(f'../../data/{model_name}_norm_stats.json', 'r') as f:
    stats = json.load(f)
    means = stats['mean']
    stds = stats['std']
ds = utils.normalize_ds(ds, means, stds, input_columns, normalize_output=True)

# Convert to tensor
def convert_to_tensor(row):
    input_ids = torch.tensor([v for k,v in row.items() if k not in output_columns and k != "DateTimeFormatted"])
    label = torch.tensor([v for k,v in row.items() if k in output_columns])
    return {"input_ids": input_ids, "label": label}


ds = ds.map(convert_to_tensor, num_proc=os.cpu_count(), remove_columns=input_columns + output_columns)
ds.set_format(type="torch", columns=["input_ids", "label"])
dates = [d for d in ds["DateTimeFormatted"]]
ds = ds.remove_columns(["DateTimeFormatted"])

data_loader = DataLoader(ds, batch_size=512, shuffle=False, num_workers=os.cpu_count())

predictions, true_values = [], []

with torch.no_grad():
    for batch in tqdm(data_loader, desc="Evaluating"):
        x = batch["input_ids"].to("cuda")
        y = batch["label"].to("cuda")

        y_pred = model(x)

        # Save Loge K Te values
        y_pred = torch.argmax(y_pred, dim=1) * 0.05 + 6 + 0.025

        # Save K Te values
        # y_pred = torch.exp(torch.argmax(y_pred, dim=1) * 0.05 + 6 + 0.025)

        y_true = y

        predictions.extend(y_pred.flatten().tolist())
        true_values.extend(y_true.flatten().tolist())

# Plot
dates = pd.to_datetime(dates)
plt.figure(figsize=(15, 8))
plt.plot(dates, true_values, label='Observed', alpha=0.6, linewidth=1)
plt.plot(dates, predictions, label='Predicted', alpha=0.6, linewidth=1)
plt.xlabel('Time')
plt.ylabel('Temperature [Loge K]')
plt.title('Electron Temperature During June 1991 Solar Storm')
plt.legend()
plt.gcf().autofmt_xdate()  # Rotate and align the tick labels
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('solar_storm.png', dpi=300, bbox_inches='tight')
plt.close()