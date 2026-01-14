# safe_loader.py
import joblib
import pickle
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree._tree import Tree
# uvicorn ml_service:app --host 0.0.0.0 --port 8001 --reload

old_path = "decision_tree_model.pkl"
new_path = "decision_tree_model_converted.pkl"

# Universal redirect unpickler
class ForceDTUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # If ANY DecisionTreeClassifier is referenced, redirect to sklearn's
        if "DecisionTree" in name or "DecisionTreeClassifier" in name:
            return DecisionTreeClassifier
        # Safest fallback: try normal resolution
        try:
            return super().find_class(module, name)
        except Exception:
            # If unknown class → force DecisionTreeClassifier
            return DecisionTreeClassifier

# Load the model with forced redirect
with open(old_path, "rb") as f:
    old_obj = ForceDTUnpickler(f).load()

print("✔ Loaded model using forced redirect")

# Extract artifact parts
if isinstance(old_obj, dict):
    old_model = old_obj.get("model")
    features = old_obj.get("features", [])
    labels = old_obj.get("label_encoder_classes", [])
else:
    old_model = old_obj
    features = []
    labels = []

print("Features found:", features)
print("Labels found:", labels)

# Extract tree from old model
old_tree = old_model.tree_

# Prepare new "nodes" array with added missing_go_to_left
children_left = old_tree.children_left
children_right = old_tree.children_right
feature = old_tree.feature
threshold = old_tree.threshold
impurity = old_tree.impurity
n_node_samples = old_tree.n_node_samples
weighted_n_node_samples = old_tree.weighted_n_node_samples

missing_go_to_left = np.zeros_like(children_left, dtype=np.uint8)

nodes = np.zeros(
    old_tree.node_count,
    dtype=[
        ('left_child', '<i8'),
        ('right_child', '<i8'),
        ('feature', '<i8'),
        ('threshold', '<f8'),
        ('impurity', '<f8'),
        ('n_node_samples', '<i8'),
        ('weighted_n_node_samples', '<f8'),
        ('missing_go_to_left', 'u1'),
    ]
)

nodes['left_child'] = children_left
nodes['right_child'] = children_right
nodes['feature'] = feature
nodes['threshold'] = threshold
nodes['impurity'] = impurity
nodes['n_node_samples'] = n_node_samples
nodes['weighted_n_node_samples'] = weighted_n_node_samples
nodes['missing_go_to_left'] = missing_go_to_left

# Create new tree compatible with your version
new_tree = Tree(
    n_features=old_tree.n_features,
    n_classes=old_tree.n_classes,
    n_outputs=old_tree.n_outputs,
)

new_tree.__setstate__({
    'max_depth': old_tree.max_depth,
    'node_count': old_tree.node_count,
    'nodes': nodes,
    'values': old_tree.value
})

# Build new model
new_model = DecisionTreeClassifier()
new_model.n_features_in_ = old_model.n_features_in_
new_model.tree_ = new_tree
new_model.classes_ = old_model.classes_
new_model.n_classes_ = old_model.n_classes_
new_model.n_outputs_ = old_model.n_outputs_

# Save modern artifact
new_artifact = {
    "model": new_model,
    "features": features,
    "label_encoder_classes": labels
}

joblib.dump(new_artifact, new_path)

print("✔ Conversion complete!")
print("✔ Saved:", new_path)
