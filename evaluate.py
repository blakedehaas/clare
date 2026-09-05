from tqdm import tqdm
import numpy as np
from matplotlib.colors import LogNorm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import models.feed_forward as models
import json
import matplotlib.pyplot as plt
import os
import datasets
import scipy
import pandas as pd
import constants
from sklearn.metrics import r2_score, mean_squared_error

import argparse

# --------- ARGUMENTS & CONFIG -----------------
parser = argparse.ArgumentParser(description="Evaluate CLARE Space Weather Models")
parser.add_argument("--model_type", type=str, default="decoder", choices=["decoder", "feed_forward"],
                    help="Model architecture: 'decoder' or 'feed_forward'")
parser.add_argument("--model_name", type=str, default="mini",
                    help="Model preset name: mini, micro, small, medium, 1_47")
parser.add_argument("--checkpoint", type=str, default=None,
                    help="Explicit path to checkpoint file")
parser.add_argument("--dataset", type=str, default="test-normal",
                    help="Dataset partition to evaluate on")
parser.add_argument("--max_eval_samples", type=int, default=5000,
                    help="Maximum number of test rows to evaluate")

def compute_metrics(predictions, true_values, entropy_list=None):
    deviations = np.array(predictions) - np.array(true_values)
    r2 = float(r2_score(true_values, predictions)) if len(true_values) > 1 else 0.0
    rmse = float(np.sqrt(mean_squared_error(true_values, predictions)))
    mae = float(np.mean(np.abs(deviations)))
    mean_entropy = float(np.mean(entropy_list)) if entropy_list is not None and len(entropy_list) > 0 else 0.0

    thresholds = [100, 200, 300, 500, 1000, 2000, 5000]
    percentages = [
        float(np.mean(np.abs(deviations) <= threshold) * 100)
        for threshold in thresholds
    ]

    relative_thresholds = [5, 10, 15, 20]
    relative_percentages = [
        float(np.mean(np.abs(deviations) / np.maximum(1e-2, np.array(true_values)) * 100 <= threshold) * 100)
        for threshold in relative_thresholds
    ]
    return {
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'mean_entropy': mean_entropy,
        'percentages': percentages,
        'relative_percentages': relative_percentages
    }

# Setting up columns
input_columns = ['Altitude', 'GCLAT', 'GCLON', 'ILAT', 'GLAT', 'GMLT', 'XXLAT', 'XXLON', 'AL_index_0', 'AL_index_1', 'AL_index_2', 'AL_index_3', 'AL_index_4', 'AL_index_5', 'AL_index_6', 'AL_index_7', 'AL_index_8', 'AL_index_9', 'AL_index_10', 'AL_index_11', 'AL_index_12', 'AL_index_13', 'AL_index_14', 'AL_index_15', 'AL_index_16', 'AL_index_17', 'AL_index_18', 'AL_index_19', 'AL_index_20', 'AL_index_21', 'AL_index_22', 'AL_index_23', 'AL_index_24', 'AL_index_25', 'AL_index_26', 'AL_index_27', 'AL_index_28', 'AL_index_29', 'AL_index_30', 'SYM_H_0', 'SYM_H_1', 'SYM_H_2', 'SYM_H_3', 'SYM_H_4', 'SYM_H_5', 'SYM_H_6', 'SYM_H_7', 'SYM_H_8', 'SYM_H_9', 'SYM_H_10', 'SYM_H_11', 'SYM_H_12', 'SYM_H_13', 'SYM_H_14', 'SYM_H_15', 'SYM_H_16', 'SYM_H_17', 'SYM_H_18', 'SYM_H_19', 'SYM_H_20', 'SYM_H_21', 'SYM_H_22', 'SYM_H_23', 'SYM_H_24', 'SYM_H_25', 'SYM_H_26', 'SYM_H_27', 'SYM_H_28', 'SYM_H_29', 'SYM_H_30', 'SYM_H_31', 'SYM_H_32', 'SYM_H_33', 'SYM_H_34', 'SYM_H_35', 'SYM_H_36', 'SYM_H_37', 'SYM_H_38', 'SYM_H_39', 'SYM_H_40', 'SYM_H_41', 'SYM_H_42', 'SYM_H_43', 'SYM_H_44', 'SYM_H_45', 'SYM_H_46', 'SYM_H_47', 'SYM_H_48', 'SYM_H_49', 'SYM_H_50', 'SYM_H_51', 'SYM_H_52', 'SYM_H_53', 'SYM_H_54', 'SYM_H_55', 'SYM_H_56', 'SYM_H_57', 'SYM_H_58', 'SYM_H_59', 'SYM_H_60', 'SYM_H_61', 'SYM_H_62', 'SYM_H_63', 'SYM_H_64', 'SYM_H_65', 'SYM_H_66', 'SYM_H_67', 'SYM_H_68', 'SYM_H_69', 'SYM_H_70', 'SYM_H_71', 'SYM_H_72', 'SYM_H_73', 'SYM_H_74', 'SYM_H_75', 'SYM_H_76', 'SYM_H_77', 'SYM_H_78', 'SYM_H_79', 'SYM_H_80', 'SYM_H_81', 'SYM_H_82', 'SYM_H_83', 'SYM_H_84', 'SYM_H_85', 'SYM_H_86', 'SYM_H_87', 'SYM_H_88', 'SYM_H_89', 'SYM_H_90', 'SYM_H_91', 'SYM_H_92', 'SYM_H_93', 'SYM_H_94', 'SYM_H_95', 'SYM_H_96', 'SYM_H_97', 'SYM_H_98', 'SYM_H_99', 'SYM_H_100', 'SYM_H_101', 'SYM_H_102', 'SYM_H_103', 'SYM_H_104', 'SYM_H_105', 'SYM_H_106', 'SYM_H_107', 'SYM_H_108', 'SYM_H_109', 'SYM_H_110', 'SYM_H_111', 'SYM_H_112', 'SYM_H_113', 'SYM_H_114', 'SYM_H_115', 'SYM_H_116', 'SYM_H_117', 'SYM_H_118', 'SYM_H_119', 'SYM_H_120', 'SYM_H_121', 'SYM_H_122', 'SYM_H_123', 'SYM_H_124', 'SYM_H_125', 'SYM_H_126', 'SYM_H_127', 'SYM_H_128', 'SYM_H_129', 'SYM_H_130', 'SYM_H_131', 'SYM_H_132', 'SYM_H_133', 'SYM_H_134', 'SYM_H_135', 'SYM_H_136', 'SYM_H_137', 'SYM_H_138', 'SYM_H_139', 'SYM_H_140', 'SYM_H_141', 'SYM_H_142', 'SYM_H_143', 'SYM_H_144', 'f107_index_0', 'f107_index_1', 'f107_index_2', 'f107_index_3', 'Kp_index']
output_columns = ['Te1']
all_columns = input_columns + output_columns


def main():  # pragma: no cover
    args, unknown = parser.parse_known_args()

    model_type = args.model_type
    model_name = args.model_name
    dataset = args.dataset
    # ----------------------------------------------



    # Device configuration (uses local RTX GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluation compute device: {device}")
    if torch.cuda.is_available():
        print(f"GPU Hardware: {torch.cuda.get_device_name(0)}")

    # Model & Checkpoint resolution
    if model_type == "decoder":
        from train_decoder import GPTConfig, GPT, SpaceWeatherMeasurementTokenizer, load_dataset
        ckpt_path = args.checkpoint if args.checkpoint else f"checkpoints/decoder_{model_name}_best.pth"
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Decoder checkpoint not found at: {ckpt_path}")
        print(f"Loading decoder checkpoint from: {ckpt_path}")
        checkpoint_data = torch.load(ckpt_path, map_location=device, weights_only=False)
        config = checkpoint_data.get("config")
        if config is None:
            cfg_path = f"configs/{model_name}.json"
            if os.path.exists(cfg_path):
                with open(cfg_path) as cf:
                    cd = json.load(cf)
                config = GPTConfig(n_layer=cd["n_layer"], n_head=cd["n_head"], n_embd=cd["n_embd"], block_size=cd["block_size"], vocab_size=cd["vocab_size"])
            else:
                config = GPTConfig(n_layer=6, n_head=6, n_embd=192, block_size=256, vocab_size=1024)
        model = GPT(config).to(device)
        model.load_state_dict(checkpoint_data["model_state_dict"])
        model.eval()
        print(f"Loaded GPT-{model_name} with {sum(p.numel() for p in model.parameters()):,} parameters.")
    else:
        input_size = len(input_columns)
        hidden_size = 2048
        output_size = 150
        model = models.FeedForwardNetwork(input_size, hidden_size, output_size).to(device)
        ckpt_path = args.checkpoint if args.checkpoint else f'checkpoints/{model_name}.pth'
        fallback_model_path = 'checkpoints/checkpoint.pth'
        if os.path.exists(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
        elif os.path.exists(fallback_model_path):
            print(f"Loading weights from fallback {fallback_model_path}")
            model.load_state_dict(torch.load(fallback_model_path, map_location=device, weights_only=False))
        model.eval()
        checkpoint_data = {}

    # Data loading & evaluation
    predictions, true_values, entropy_list, times = [], [], [], []

    if model_type == "decoder":
        tokenizer = SpaceWeatherMeasurementTokenizer(feature_names=[], target_name="Te1", vocab_size=config.vocab_size)
        tokenizer.load_stats("checkpoints/decoder_tokenizer.json")

        print("Loading test space weather data...")
        raw_df = load_dataset()
        if "DateFormatted" in raw_df.columns and "TimeFormatted" in raw_df.columns and "DateTimeFormatted" not in raw_df.columns:
            raw_df["DateTimeFormatted"] = pd.to_datetime(raw_df["DateFormatted"].dt.strftime('%Y-%m-%d') + ' ' + raw_df['TimeFormatted'].astype(str), errors='coerce')

        split_idx = int(0.80 * len(raw_df))
        test_df = raw_df.iloc[split_idx:].copy()
        if len(test_df) > args.max_eval_samples:
            test_df = test_df.iloc[:args.max_eval_samples].copy()

        print(f"Evaluating across {len(test_df):,} test observations...")
        feature_names = [col for col in tokenizer.feature_names if col in test_df.columns]
        target_name = tokenizer.target_name

        batch_size = 64
        block_size = config.block_size

        with torch.no_grad():
            for i in tqdm(range(0, len(test_df), batch_size), desc="Evaluating Batches"):
                batch_slice = test_df.iloc[i : i + batch_size]
                b_size = len(batch_slice)
                batch_contexts, batch_true_k, batch_times = [], [], []

                for row_idx, (_, row) in enumerate(batch_slice.iterrows()):
                    param_tokens = [tokenizer.encode_measurement(col, row[col]) for col in feature_names]
                    if len(param_tokens) > block_size - 1:
                        param_tokens = param_tokens[-(block_size - 1):]
                    batch_contexts.append(torch.tensor(param_tokens, dtype=torch.long))
                    batch_true_k.append(float(row[target_name]))
                    dt = row.get("DateTimeFormatted", i + row_idx)
                    batch_times.append(str(dt))

                max_len = max(len(c) for c in batch_contexts)
                padded = torch.zeros((b_size, max_len), dtype=torch.long, device=device)
                for b_i, ctx in enumerate(batch_contexts):
                    padded[b_i, :len(ctx)] = ctx

                logits, _ = model(padded)
                last_logits = logits[:, -1, :]

                probs = F.softmax(last_logits, dim=-1)
                entropies = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1).cpu().numpy()
                entropy_list.extend(entropies.tolist())

                pred_toks = torch.argmax(last_logits, dim=-1).cpu().numpy()
                for p_tok in pred_toks:
                    predictions.append(tokenizer.decode_token(target_name, int(p_tok)))

                true_values.extend(batch_true_k)
                times.extend(batch_times)
    else:
        # Load dataset
        test_ds = datasets.Dataset.load_from_disk(f"dataset/processed_dataset/{dataset}")
        test_ds = test_ds.remove_columns(['Ne1', 'Pv1', 'Te2', 'Ne2', 'Pv2', 'Te3', 'Ne3', 'Pv3', 'I1', 'I2', 'I3'])

        def normalize_batch(batch):
            for col, norm_func in constants.NORMALIZATIONS.items():
                batch[col] = norm_func(batch[col])
            return batch

        test_ds = test_ds.map(normalize_batch, batched=True, batch_size=10000, num_proc=os.cpu_count())

        columns_to_normalize = [col for col in input_columns if col.startswith('AL_index') or col.startswith('SYM_H') or col.startswith('f107_index')]
        index_groups = {
            'AL_index': [col for col in columns_to_normalize if col.startswith('AL_index')],
            'SYM_H': [col for col in columns_to_normalize if col.startswith('SYM_H')],
            'f107_index': [col for col in columns_to_normalize if col.startswith('f107_index')]
        }
        means, stds = {}, {}
        stats_file = f'checkpoints/{model_name}_norm_stats.json'
        if os.path.exists(stats_file):
            with open(stats_file, 'r') as f:
                stats = json.load(f)
                means = stats['mean']
                stds = stats['std']

        group_cols = [col for cols in index_groups.values() for col in cols]
        def normalize_group(batch):
            for col in group_cols:
                group_name = '_'.join(col.split('_')[:-1]) if col.split('_')[-1].isdigit() else col
                values = np.array(batch[col], dtype=np.float32)
                batch[col] = (values - means[group_name]) / stds[group_name]
            return batch

        test_ds = test_ds.map(normalize_group, batched=True, batch_size=10000, num_proc=os.cpu_count())

        def convert_to_tensor(row):
            input_ids = torch.tensor([v for k,v in row.items() if k in input_columns])
            label = torch.tensor([v for k,v in row.items() if k in output_columns])
            return {
                "input_ids": input_ids, 
                "label": label,
                "DateTimeFormatted": row['DateTimeFormatted']
            }
        test_ds = test_ds.map(convert_to_tensor, num_proc=os.cpu_count(), remove_columns=all_columns)

        def custom_collate(batch):
            input_ids = torch.stack([torch.tensor(item['input_ids']) for item in batch])
            labels = torch.stack([torch.tensor(item['label']) for item in batch])
            datetimes = [item['DateTimeFormatted'] for item in batch]
            return {
                'input_ids': input_ids,
                'label': labels,
                'DateTimeFormatted': datetimes
            }

        test_loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=os.cpu_count(), collate_fn=custom_collate)

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                x = batch["input_ids"].to(device)
                y = batch["label"].to(device)

                logits = model(x)
                softmaxed = torch.softmax(logits, dim=1)
                entropy = -torch.sum(softmaxed * torch.log(softmaxed + 1e-10), dim=1).cpu().numpy()
                entropy_list.extend(entropy)

                y_pred = torch.argmax(logits, dim=1) * 100 + 50
                predictions.extend(y_pred.flatten().tolist())
                true_values.extend(y.flatten().tolist())
                times.extend(batch['DateTimeFormatted'])

    deviations = [pred - true for pred, true in zip(predictions, true_values)]
    model_prefix = f"decoder_{model_name}" if model_type == "decoder" else model_name

    # Calculate R^2 score
    r2 = r2_score(true_values, predictions)
    print(f"\nR^2 Score: {r2:.4f}")

    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(true_values, predictions))
    print(f"RMSE: {rmse:.4f} K")

    # Calculate entropy metrics
    mean_entropy = np.mean(entropy_list)
    print(f"Mean entropy across test set: {mean_entropy:.4f}")

    # Calculate percentages within specified absolute deviations
    thresholds = [100, 200, 300, 500, 1000, 2000, 5000]
    percentages = [
        sum(abs(dev) <= threshold for dev in deviations) / len(deviations) * 100
        for threshold in thresholds
    ]

    # Calculate percentages within specified relative deviations
    relative_thresholds = [5, 10, 15, 20]
    relative_percentages = [
        sum(abs(dev) / max(1e-2, true) * 100 <= threshold for dev, true in zip(deviations, true_values)) / len(deviations) * 100
        for threshold in relative_thresholds
    ]

    # Plot histogram
    plt.figure(figsize=(12, 8))
    plt.hist(deviations, bins=50, edgecolor='black', color="#3182BD", alpha=0.85)
    plt.xlabel('Deviation from Ground Truth (Te_pred - Te_obs) [K]')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Model Predictions Deviation ({model_prefix.upper()})')

    # Add text box with percentages and metrics
    text = "\n".join([
        f"R² Score: {r2:.4f}",
        f"RMSE: {rmse:.4f}",
        f"Mean entropy: {mean_entropy:.4f}",
        "\n"
    ] + [
        f"Within {threshold}: {percentage:.2f}%"
        for threshold, percentage in zip(thresholds, percentages)
    ] + ["\n"] + [
        f"Within {threshold}%: {percentage:.2f}%"
        for threshold, percentage in zip(relative_thresholds, relative_percentages)
    ])
    print("\n" + text + "\n")
    plt.text(0.95, 0.95, text, transform=plt.gca().transAxes, 
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    hist_plot_path = f'./checkpoints/{model_prefix}_{dataset}_plot.png'
    if os.path.exists(hist_plot_path):
        try: os.remove(hist_plot_path)
        except Exception: pass
    plt.savefig(hist_plot_path)
    plt.close()
    print(f"Saved prediction deviation plot to: {hist_plot_path}")

    # Plot absolute deviation vs ground truth
    plt.figure(figsize=(10, 8))
    h = plt.hist2d(true_values, deviations, bins=100, norm=LogNorm(), cmap='viridis')
    plt.colorbar(h[3], label='Obs#')

    plt.xlabel('Te$_{obs}$ [K]')
    plt.ylabel('Te$_{model}$ - Te$_{obs}$ [K]')
    plt.title(f'Model Deviation vs Ground Truth ({model_prefix.upper()})')

    bin_means, bin_edges, _ = scipy.stats.binned_statistic(true_values, deviations, statistic='mean', bins=50)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    plt.plot(bin_centers, bin_means, 'r-', linewidth=2, label='Mean Deviation')
    plt.legend()

    mean_deviation = np.mean(deviations)
    print(f"Mean Deviation: {mean_deviation:.3f} K")

    plt.tight_layout()
    dev_plot_path = f'./checkpoints/{model_prefix}_{dataset}_deviation_plot.png'
    if os.path.exists(dev_plot_path):
        try: os.remove(dev_plot_path)
        except Exception: pass
    plt.savefig(dev_plot_path, dpi=300)
    plt.close()
    print(f"Saved 2D deviation density plot to: {dev_plot_path}")

    # Visualization 1: Epoch vs Test/Val/Train Loss Curve (@Xiangning Chu)
    try:
        from visualizations import TrainingLossVisualizer
        loss_history = checkpoint_data.get("loss_history")
        epoch_loss_path = f'./checkpoints/{model_prefix}_epoch_loss_curve.png'
        epoch_test_loss_path = f'./checkpoints/{model_prefix}_epoch_test_loss_curve.png'
        if loss_history and len(loss_history.get("steps", [])) > 0:
            loss_vis = TrainingLossVisualizer(history=loss_history)
            loss_vis.generate_plot(
                title=f"Model {model_prefix.upper()}: Training & Validation Loss vs Epochs",
                save_path=epoch_loss_path
            )
            print(f"Saved epoch loss curve plot to {epoch_loss_path}")
            loss_vis.generate_epoch_test_loss_plot(
                title=f"Model {model_prefix.upper()}: Test Loss vs Epochs (Best Val Model Marked)",
                save_path=epoch_test_loss_path
            )
            print(f"Saved epoch test loss curve plot to {epoch_test_loss_path}")
        elif os.path.exists(f'./checkpoints/{model_prefix}_loss_curve.png') and not os.path.exists(epoch_loss_path):
            import shutil
            shutil.copyfile(f'./checkpoints/{model_prefix}_loss_curve.png', epoch_loss_path)
            print(f"Copied loss curve plot to {epoch_loss_path}")
    except Exception as e:
        print(f"Note: Could not generate epoch loss curve plot: {e}")

    # Visualization 2: Sandwiched Test Block Visualization Suite (@Michael)
    try:
        from visualizations import SandwichedBlockVisualizer
        df_eval = pd.DataFrame({
            'DateTimeFormatted': times,
            'Te1': true_values,
            'Te1_pred': predictions,
            'split': 'test'
        })
        if len(df_eval) >= 150:
            sand_vis = SandwichedBlockVisualizer()
            block_size = min(150, len(df_eval) // 3)
            candidate_blocks = []
            n_eval = len(df_eval)
            num_candidates = min(50, n_eval // (3 * block_size))
            num_candidates = max(5, num_candidates)

            for k in range(num_candidates):
                start_k = k * block_size
                end_k = start_k + 3 * block_size
                if end_k > n_eval:
                    break
                sub_chunk = df_eval.iloc[start_k:end_k]
                y_t = sub_chunk['Te1'].values
                y_p = sub_chunk['Te1_pred'].values
                test_yt = y_t[block_size:2*block_size]
                test_yp = y_p[block_size:2*block_size]
                r2_val = float(r2_score(test_yt, test_yp)) if len(test_yt) > 1 else 0.0
                rmse_val = float(np.sqrt(mean_squared_error(test_yt, test_yp)))
                mae_val = float(mean_absolute_error(test_yt, test_yp))

                candidate_blocks.append({
                    'candidate_idx': k,
                    'y_true': y_t,
                    'y_pred': y_p,
                    'train1_len': block_size,
                    'test_len': block_size,
                    'train2_len': len(y_t) - 2 * block_size,
                    'r2': r2_val,
                    'rmse': rmse_val,
                    'mae': mae_val,
                })

            if candidate_blocks:
                ranked = sorted(candidate_blocks, key=lambda b: b['r2'])
                best_cand = ranked[-1]
                worst_cand = ranked[0]
                median_cand = ranked[len(ranked) // 2]
                rng = np.random.RandomState(42)
                random_cand = candidate_blocks[rng.randint(0, len(candidate_blocks))]

                def plot_cand(cand, title, save_p):
                    n_c = len(cand['y_true'])
                    df_p = pd.DataFrame({'Te1': cand['y_true'], 'Te1_pred': cand['y_pred']})
                    m1 = np.zeros(n_c, dtype=bool); m1[:cand['train1_len']] = True
                    m2 = np.zeros(n_c, dtype=bool); m2[cand['train1_len']:cand['train1_len']+cand['test_len']] = True
                    m3 = np.zeros(n_c, dtype=bool); m3[cand['train1_len']+cand['test_len']:] = True
                    sand_vis.generate_plot(df_p, m1, m2, m3, title=title, save_path=save_p)

                p_rand = f'./checkpoints/{model_prefix}_sandwiched_random.png'
                p_best = f'./checkpoints/{model_prefix}_sandwiched_best.png'
                p_worst = f'./checkpoints/{model_prefix}_sandwiched_worst.png'
                p_med = f'./checkpoints/{model_prefix}_sandwiched_median.png'
                p_mean = f'./checkpoints/{model_prefix}_sandwiched_mean_all_blocks.png'

                plot_cand(random_cand, f"Model {model_prefix.upper()}: Random Sandwiched Test Block (R² = {random_cand['r2']:.3f})", p_rand)
                plot_cand(best_cand, f"Model {model_prefix.upper()}: Best Sandwiched Test Block (R² = {best_cand['r2']:.3f})", p_best)
                plot_cand(worst_cand, f"Model {model_prefix.upper()}: Worst Sandwiched Test Block (R² = {worst_cand['r2']:.3f})", p_worst)
                plot_cand(median_cand, f"Model {model_prefix.upper()}: Median Sandwiched Test Block (R² = {median_cand['r2']:.3f})", p_med)

                sand_vis.generate_mean_sandwiched_plot(
                    candidate_blocks,
                    title=f"Model {model_prefix.upper()}: Mean Performance Over All Sandwiched Blocks (N={len(candidate_blocks)})",
                    save_path=p_mean
                )
                # Also copy to standard sandwiched_block.png
                import shutil
                shutil.copyfile(p_rand, f'./checkpoints/{model_prefix}_sandwiched_block.png')
                print(f"Saved complete 5-case Sandwiched Suite for {model_prefix.upper()}")
    except Exception as e:
        print(f"Note: Could not generate sandwiched block evaluation plot: {e}")

    print("\nEvaluation and all visualizations successfully generated!")

    print("\nEvaluation and all visualizations successfully generated!")


if __name__ == "__main__":  # pragma: no cover
    main()
