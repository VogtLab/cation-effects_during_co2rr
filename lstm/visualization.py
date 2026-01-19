import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import os
from sklearn.metrics import mean_squared_error, r2_score


def setup_plotting():
    mpl.use('SVG')
    mpl.rcParams['svg.fonttype'] = 'none'
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_context("paper", font_scale=1.5)


def plot_correlation_matrix(df, title, output_dir=None):
    plt.figure(figsize=(14, 12), dpi=300)
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.color_palette("Blues", as_cmap=True)
    ax = sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt='.2f',
        cmap=cmap,
        vmin=-1,
        vmax=1,
        annot_kws={"size": 10},
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Correlation Coefficient"}
    )

    plt.title(title, fontsize=18, pad=20)
    plt.xticks(fontsize=12, rotation=45, ha='right')
    plt.yticks(fontsize=12)
    plt.tight_layout()

    if output_dir:
        file_name = f"{title.replace(' ', '_')}"
        plt.savefig(os.path.join(output_dir, f"{file_name}.png"), bbox_inches='tight')
        plt.savefig(os.path.join(output_dir, f"{file_name}.svg"), bbox_inches='tight', format='svg')

    plt.close()


def plot_feature_importance(importance, feature_names, title, output_dir=None):
    plt.figure(figsize=(14, 10), dpi=300)

    sorted_idx = np.argsort(importance)
    sorted_importance = importance[sorted_idx]
    sorted_names = [feature_names[i] for i in sorted_idx]

    colors = plt.cm.Blues(np.linspace(0.2, 0.8, len(sorted_idx)))
    bars = plt.barh(range(len(sorted_idx)), sorted_importance, color=colors)
    plt.yticks(range(len(sorted_idx)), sorted_names, fontsize=14)
    plt.xlabel('Feature Importance', fontsize=16)
    plt.title(title, fontsize=18, pad=20)

    for i, v in enumerate(sorted_importance):
        plt.text(v + 0.01*max(importance), i, f"{v:.2e}", va='center', fontsize=12)

    plt.grid(axis='x', linestyle='--', alpha=0.7)

    plt.tight_layout()

    if output_dir:
        file_name = f"{title.replace(' ', '_')}"
        plt.savefig(os.path.join(output_dir, f"{file_name}.png"), bbox_inches='tight')
        plt.savefig(os.path.join(output_dir, f"{file_name}.svg"), bbox_inches='tight', format='svg')

    plt.close()


def plot_predictions(y_true, y_pred, title, output_dir=None):
    plt.figure(figsize=(12, 10), dpi=300)

    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)

    scatter = plt.scatter(y_true, y_pred, c=y_true, cmap='Blues',
                         s=80, alpha=0.7, edgecolor='white', linewidth=0.5)

    cbar = plt.colorbar(scatter)
    cbar.set_label('Actual Values', fontsize=14)

    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='Perfect Prediction')

    plt.xlabel('Actual', fontsize=16)
    plt.ylabel('Predicted', fontsize=16)
    plt.title(title, fontsize=18, pad=20)

    textstr = f'$R^2 = {r2:.4f}$\n$MSE = {mse:.4f}$'
    props = dict(boxstyle='round', facecolor='white', alpha=0.7)
    plt.annotate(textstr, xy=(0.05, 0.95), xycoords='axes fraction',
                fontsize=14, bbox=props, verticalalignment='top')
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()

    if output_dir:
        file_name = f"{title.replace(' ', '_')}"
        plt.savefig(os.path.join(output_dir, f"{file_name}.png"), bbox_inches='tight')
        plt.savefig(os.path.join(output_dir, f"{file_name}.svg"), bbox_inches='tight', format='svg')

    plt.close()
    return r2, mse


def plot_cation_predictions(y_true, y_pred, cations, title, output_dir=None):
    plt.figure(figsize=(12, 10), dpi=300)

    cation_colors = {'Na': '#1f77b4', 'K': '#2ca02c', 'Cs': '#d62728', 'Li': '#9467bd'}

    for cation in set(cations):
        mask = np.array(cations) == cation
        if sum(mask) > 0:
            plt.scatter(
                y_true[mask],
                y_pred[mask],
                color=cation_colors.get(cation, 'gray'),
                label=f"{cation} (R²={r2_score(y_true[mask], y_pred[mask]):.4f})",
                alpha=0.7,
                s=80,
                edgecolor='white',
                linewidth=0.5
            )

    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2)

    overall_r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    textstr = f'Overall $R^2 = {overall_r2:.4f}$\nOverall $MSE = {mse:.4f}$'
    props = dict(boxstyle='round', facecolor='white', alpha=0.7)
    plt.annotate(textstr, xy=(0.05, 0.95), xycoords='axes fraction',
                fontsize=14, bbox=props, verticalalignment='top')

    plt.xlabel('Actual', fontsize=16)
    plt.ylabel('Predicted', fontsize=16)
    plt.title(title, fontsize=18, pad=20)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    if output_dir:
        file_name = f"{title.replace(' ', '_')}"
        plt.savefig(os.path.join(output_dir, f"{file_name}.png"), bbox_inches='tight')
        plt.savefig(os.path.join(output_dir, f"{file_name}.svg"), bbox_inches='tight', format='svg')

    plt.close()
    return overall_r2, mse


def plot_batch_predictions(y_true, y_pred, batches, title, output_dir=None):
    plt.figure(figsize=(12, 10), dpi=300)

    batch_colors = {
        'batch_1': '#1f77b4',
        'batch_2': '#ff7f0e',
        'batch_3': '#2ca02c',
        'batch_4': '#d62728',
        'batch_5': '#9467bd'
    }

    for batch in set(batches):
        mask = np.array(batches) == batch
        if sum(mask) > 0:
            plt.scatter(
                y_true[mask],
                y_pred[mask],
                color=batch_colors.get(batch, 'gray'),
                label=f"{batch} (R²={r2_score(y_true[mask], y_pred[mask]):.4f})",
                alpha=0.7,
                s=80,
                edgecolor='white',
                linewidth=0.5
            )

    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2)

    overall_r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    textstr = f'Overall $R^2 = {overall_r2:.4f}$\nOverall $MSE = {mse:.4f}$'
    props = dict(boxstyle='round', facecolor='white', alpha=0.7)
    plt.annotate(textstr, xy=(0.05, 0.95), xycoords='axes fraction',
                fontsize=14, bbox=props, verticalalignment='top')

    plt.xlabel('Actual', fontsize=16)
    plt.ylabel('Predicted', fontsize=16)
    plt.title(title, fontsize=18, pad=20)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    if output_dir:
        file_name = f"{title.replace(' ', '_')}"
        plt.savefig(os.path.join(output_dir, f"{file_name}.png"), bbox_inches='tight')
        plt.savefig(os.path.join(output_dir, f"{file_name}.svg"), bbox_inches='tight', format='svg')

    plt.close()
    return overall_r2, mse


def plot_training_history(train_losses, val_losses, target_peak, output_dir=None):
    plt.figure(figsize=(12, 8), dpi=300)
    plt.plot(train_losses, color='#9467bd', linewidth=2.5, label='Training Loss')
    plt.plot(val_losses, color='#e377c2', linewidth=2.5, label='Validation Loss')

    plt.fill_between(range(len(train_losses)), train_losses, alpha=0.2, color='#9467bd')
    plt.fill_between(range(len(val_losses)), val_losses, alpha=0.2, color='#e377c2')

    plt.title(f'Training History: {target_peak}', fontsize=18, pad=20)
    plt.xlabel('Epoch', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.legend(fontsize=14, frameon=True, facecolor='white', framealpha=0.9)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    if output_dir:
        plt.savefig(os.path.join(output_dir, 'Training_History.png'), bbox_inches='tight')
        plt.savefig(os.path.join(output_dir, 'Training_History.svg'), bbox_inches='tight', format='svg')

    plt.close()
