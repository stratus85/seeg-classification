import torch
from torchviz import make_dot
from torchview import draw_graph
from torchinfo import summary
from MSCNNBiLSTM2 import create_model
from datetime import datetime
from preprocessing import save_or_preprocess, dataset_to_numpy, get_electrode_groups

pn = 2
print("=" * 80)
print(f"ResNet - Patient {pn}")
print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)

# Load data
train_set, valid_set, test_set = save_or_preprocess(pn, cache_dir='../Preprocessed')

X_train, y_train = dataset_to_numpy(train_set)
X_test, y_test = dataset_to_numpy(test_set)

n_channels = X_train.shape[1]
n_times = X_train.shape[2]

print(f"Input: {n_channels} channels x {n_times} time points")
print(f"Training: {len(X_train)} samples")
print(f"Test: {len(X_test)} samples")

electrode_groups = get_electrode_groups(pn)

# Create model
model = create_model(n_channels, n_times, n_classes=5)

dummy = torch.randn(32, 121, 400)
output = model(dummy)

# Generate DOT file and render as high‑res PNG
#dot = make_dot(output, params=dict(model.named_parameters()))
#dot.render("stscnn", format="pdf", view = True)

# Generate graph – this will show only the modules
#graph = draw_graph(model, input_data=dummy, expand_nested=True)
#graph.visual_graph.node_attr["fontname"] = "Helvetica"
#graph.visual_graph.render("stscnn_clean", format="png", view=True)
summary(model, input_size=(32, 121, 400), depth=4)
