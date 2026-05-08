import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
from preprocessing import save_or_preprocess, dataset_to_numpy
from MSCNNBiLSTM2 import SEEGNet, create_model

# -------------------------------
# Grad-CAM for 2D feature maps (C, T)
# -------------------------------
class GradCAM1D:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self._register_hook()
    
    def _register_hook(self):
        def forward_hook(module, inp, out):
            self.activations = out   # (batch, C, T)
        self.target_layer.register_forward_hook(forward_hook)
    
    def compute_heatmap(self, input_tensor, class_idx=None):
        # Temporarily set to train mode for LSTM backward
        original_mode = self.model.training
        self.model.train()
        try:
            self.model.zero_grad()
            output = self.model(input_tensor)
            if class_idx is None:
                class_idx = output.argmax().item()
            score = output[:, class_idx]
            gradients = torch.autograd.grad(score, self.activations, retain_graph=False)[0]  # (B, C, T)
            # Saliency = |grad * activation|, averaged over batch (batch=1)
            saliency = (gradients * self.activations).abs().mean(dim=0).cpu().detach().numpy()  # (C, T)
            # Normalize
            saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
            return saliency, class_idx
        finally:
            self.model.train(original_mode)

def plot_heatmap_with_sidebar(heatmap, vmin, vmax, save_path, channel_labels=False):
    """
    heatmap: 2D array (n_features, n_time)
    Creates a figure with main heatmap on left and horizontal bar chart on right.
    """
    n_features, n_time = heatmap.shape
    # Compute cumulative saliency per feature (sum over time)
    cum_saliency = heatmap.sum(axis=1)   # shape (n_features,)
    # Normalize bar widths relative to max (for visibility)
    max_cum = cum_saliency.max()
    bar_widths = (cum_saliency / max_cum) * 1.0   # scale to max width 1.0 (will be adjusted)

    fig = plt.figure(figsize=(10, 8))
    # Use GridSpec: left for heatmap, narrow gap, right for bar chart
    gs = fig.add_gridspec(1, 2, width_ratios=[4, 0.8], wspace=0.05)
    ax_heatmap = fig.add_subplot(gs[0])
    ax_bar = fig.add_subplot(gs[1])

    # Draw heatmap
    im = ax_heatmap.imshow(heatmap, cmap='jet', aspect='auto', vmin=vmin, vmax=vmax)
    ax_heatmap.set_xlabel('Time (convolutional axis)', fontsize=20)
    ax_heatmap.set_ylabel('CNN feature channel', fontsize=20)

    # Draw horizontal bars (each bar's width = normalized cumulative saliency)
    # Bar chart: y positions from 0 to n_features-1
    y_pos = np.arange(n_features)
    # Use 'hspan' or horizontal bar: barh takes width as first argument
    ax_bar.barh(y_pos, bar_widths, height=0.8, color='darkred', alpha=0.7)
    ax_bar.set_xlim(0, 1.0)
    ax_bar.set_ylim(-0.5, n_features-0.5)
    ax_bar.set_yticks([])          # hide y ticks (redundant with heatmap y-axis)
    ax_bar.set_xlabel('Relative\nimportance', fontsize=12)
    ax_bar.invert_yaxis()          # match heatmap's y-axis direction
    ax_bar.spines['top'].set_visible(False)
    ax_bar.spines['right'].set_visible(False)
    ax_bar.spines['bottom'].set_visible(True)
    ax_bar.spines['left'].set_visible(False)

    # Colorbar placed to the right of the bar chart? Actually we can place it at the bottom or keep separate.
    # For simplicity, add a colorbar at the bottom of the figure.
    cbar = fig.colorbar(im, ax=[ax_heatmap, ax_bar], orientation='horizontal', pad=0.15, aspect=50, label='Saliency')
    cbar.ax.xaxis.label.set_size(16)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# -------------------------------
# Main: load model, data, run Grad-CAM
# -------------------------------
if __name__ == '__main__':
    pn = 41
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    out_dir = f'../GradCAM results/MSC-BiLSTM-P{pn}'
    os.makedirs(out_dir, exist_ok=True)

    # 1. Load data
    print(f"Loading data for pn={pn}...")
    train_set, valid_set, test_set = save_or_preprocess(pn, cache_dir='../Preprocessed')
    X_train, y_train = dataset_to_numpy(train_set)
    X_test, y_test = dataset_to_numpy(test_set)
    n_channels, n_times = X_train.shape[1], X_train.shape[2]
    print(f"Test samples: {len(X_test)}, shape: {n_channels}×{n_times}")

    # 2. Create model and load weights
    model = create_model(n_channels, n_times, n_classes=5).to(device)
    model_path = '../SEEGNet2 results/P41/seegnet.pt'   # adjust path if needed
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Loaded model from {model_path}")

    # 3. Evaluate test accuracy
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_test_tensor = torch.LongTensor(y_test).to(device)
    with torch.no_grad():
        logits = model(X_test_tensor)
        preds = logits.argmax(dim=1)
        acc = (preds == y_test_tensor).float().mean().item()
    print(f"Test accuracy: {acc:.4f}")

    # 4. Target layer: aggregation (output of AggregationLayer)
    target_layer = model.aggregation
    gradcam = GradCAM1D(model, target_layer)

    # 5. Collect saliency maps for correctly classified test samples
    preds_np = preds.cpu().numpy()
    class_maps = {c: [] for c in range(5)}
    for idx in range(len(X_test)):
        if preds_np[idx] != y_test[idx]:
            continue
        sample = torch.FloatTensor(X_test[idx]).unsqueeze(0).to(device)  # (1, C, T)
        heatmap, _ = gradcam.compute_heatmap(sample)
        class_maps[y_test[idx]].append(heatmap)

    # 6. Compute average per class
    avg_maps = {}
    for c in range(5):
        if class_maps[c]:
            avg_maps[c] = np.mean(class_maps[c], axis=0)
        else:
            avg_maps[c] = None
            print(f"Class {c}: no correct samples")

    # 7. Global scaling across classes
    all_heatmaps = [avg_maps[c] for c in range(5) if avg_maps[c] is not None]
    if all_heatmaps:
        vmin = np.min([h.min() for h in all_heatmaps])
        vmax = np.max([h.max() for h in all_heatmaps])
    else:
        vmin, vmax = 0, 1
    print(f"Global saliency range: [{vmin:.4f}, {vmax:.4f}]")

    # 8. Save individual class heatmaps (no titles, only heatmap)
    individual_paths = []
    for c in range(5):
        if avg_maps[c] is None:
            continue
        out_path = os.path.join(out_dir, f'class_{c+1}_saliency.png')
        plot_heatmap_with_sidebar(avg_maps[c], vmin, vmax, out_path)

    # 9. Combine all five into one figure with big class titles
    if len(individual_paths) == 5:
        fig_comb, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        for i, path in enumerate(individual_paths):
            img = plt.imread(path)
            axes[i].imshow(img)
            axes[i].axis('off')
            axes[i].set_title(f'Class {i+1}', fontsize=16)
        axes[5].axis('off')
        plt.tight_layout()
        combined_path = os.path.join(out_dir, 'combined_saliency_classes_1-5.png')
        plt.savefig(combined_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Combined figure saved: {combined_path}")

    print(f"\nAll results saved in {out_dir}")