import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from preprocessing import save_or_preprocess, dataset_to_numpy
from BiLSTM3 import MultiHeadSelfAttention, ImprovedBiLSTMWithAttention, create_model

class GradCAM_1D:
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
        original_mode = self.model.training
        self.model.train()
        try:
            self.model.zero_grad()
            output = self.model(input_tensor)
            if class_idx is None:
                class_idx = output.argmax().item()
            score = output[:, class_idx]
            gradients = torch.autograd.grad(score, self.activations, retain_graph=False)[0]  # (B, C, T)
            # Compute saliency map: |grad * activation|, then average over batch
            saliency = (gradients * self.activations).abs().mean(dim=0).cpu().detach().numpy()  # (C, T)
            # Normalize
            saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
            return saliency, class_idx
        finally:
            self.model.train(original_mode)

# -------------------------------
# Main
# -------------------------------
if __name__ == '__main__':
    pn = 41
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    out_dir = f'../GradCAM results/CNN-BiLSTM-AM-P{pn}'
    os.makedirs(out_dir, exist_ok=True)

    # Load data
    print(f"Loading data for pn={pn}...")
    train_set, valid_set, test_set = save_or_preprocess(pn, cache_dir='../Retired preprocessing')
    X_train, y_train = dataset_to_numpy(train_set)
    X_test, y_test = dataset_to_numpy(test_set)
    n_channels, n_times = X_train.shape[1], X_train.shape[2]
    print(f"Test samples: {len(X_test)}, shape: {n_channels}×{n_times}")

    # Create model and load weights
    model = create_model(n_channels, n_times, n_classes=5, use_attention=True).to(device)
    model_path = '../BiLSTM results/improved_attention3/P41/improved_with_attention.pt'
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Loaded model from {model_path}")

    # Evaluate test accuracy
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_test_tensor = torch.LongTensor(y_test).to(device)
    with torch.no_grad():
        logits = model(X_test_tensor)
        preds = logits.argmax(dim=1)
        acc = (preds == y_test_tensor).float().mean().item()
    print(f"Test accuracy: {acc:.4f}")

    # Target layer: channel_reduction (outputs (batch, reduced_channels, time))
    target_layer = model.channel_reduction
    gradcam = GradCAM_1D(model, target_layer)

    # Collect saliency maps for correctly classified test samples
    preds_np = preds.cpu().numpy()
    class_maps = {c: [] for c in range(5)}
    for idx in range(len(X_test)):
        if preds_np[idx] != y_test[idx]:
            continue
        sample = torch.FloatTensor(X_test[idx]).unsqueeze(0).to(device)  # (1, C, T)
        heatmap, _ = gradcam.compute_heatmap(sample)
        class_maps[y_test[idx]].append(heatmap)

    # Compute average per class
    avg_maps = {}
    for c in range(5):
        if class_maps[c]:
            avg_maps[c] = np.mean(class_maps[c], axis=0)
        else:
            avg_maps[c] = None
            print(f"Class {c}: no correct samples")

    # Find global min and max across all classes
    all_heatmaps = [avg_maps[c] for c in range(5) if avg_maps[c] is not None]
    if all_heatmaps:
        vmin = np.min([h.min() for h in all_heatmaps])
        vmax = np.max([h.max() for h in all_heatmaps])
    else:
        vmin, vmax = 0, 1
    print(f"Global saliency range: [{vmin:.4f}, {vmax:.4f}]")

    # Plot each class heatmap (2D: reduced_channels × time)
    individual_paths = []
    for c in range(5):
        if avg_maps[c] is None:
            continue
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(avg_maps[c], cmap='jet', aspect='auto', vmin=vmin, vmax=vmax)
        ax.set_xlabel('Time (convolutional axis)', fontsize=20)
        ax.set_ylabel('Channels (convolutional axis)', fontsize=20)
        cbar = plt.colorbar(im, ax=ax, label='Saliency')
        cbar.ax.yaxis.label.set_size(20)
        cbar.ax.tick_params(labelsize=16)
        plt.tight_layout()
        out_path = os.path.join(out_dir, f'class_{c+1}_saliency.png')
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        individual_paths.append(out_path)

    # Combine into one figure
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