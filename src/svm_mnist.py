# src/svm_mnist.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score
from torchvision import datasets, transforms
import os

# Ensure results folder exists
os.makedirs("../results", exist_ok=True)

# Load MNIST images
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor()
])

train_data = datasets.ImageFolder(root='../MNIST-full/train', transform=transform)
test_data = datasets.ImageFolder(root='../MNIST-full/test', transform=transform)

# Convert to numpy arrays
def dataset_to_numpy(dataset):
    X, y = [], []
    for img, label in dataset:
        X.append(img.view(-1).numpy())
        y.append(label)
    return np.array(X), np.array(y)

X_train, y_train = dataset_to_numpy(train_data)
X_test, y_test = dataset_to_numpy(test_data)

# Split validation set from training
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=10000, random_state=42)

# Define parameter grid for cross-validation
param_grid = [
    {'kernel': ['linear'], 'C': [0.1, 1, 10]},
    {'kernel': ['rbf'], 'C': [1, 10], 'gamma': [0.001, 0.01, 0.1]}
]

print("🔍 Starting Grid Search on SVM...")
clf = GridSearchCV(SVC(), param_grid, cv=3, n_jobs=-1, verbose=2, return_train_score=True)
clf.fit(X_train, y_train)

print("✅ Best Parameters:", clf.best_params_)
print("✅ Best CV Accuracy:", clf.best_score_)

# Evaluate on validation set
y_pred_val = clf.best_estimator_.predict(X_val)
val_acc = accuracy_score(y_val, y_pred_val)
print(f"📊 Validation Accuracy: {val_acc:.4f}")

# Evaluate on test set
y_pred_test = clf.best_estimator_.predict(X_test)
test_acc = accuracy_score(y_test, y_pred_test)
print(f"🏁 Test Accuracy: {test_acc:.4f}")

# Save summary results
with open("../results/svm_results.txt", "w") as f:
    f.write(f"Best params: {clf.best_params_}\n")
    f.write(f"Validation Accuracy: {val_acc}\n")
    f.write(f"Test Accuracy: {test_acc}\n")

# Save cross-validation results for plotting
cv_results = clf.cv_results_
np.save("../results/svm_cv_results.npy", cv_results)  # Save all CV info

# Plot accuracy vs hyperparameters for each kernel
import pandas as pd
results_df = pd.DataFrame(cv_results)

# Linear kernel
linear_results = results_df[results_df['param_kernel'] == 'linear']
plt.figure()
plt.plot(linear_results['param_C'], linear_results['mean_test_score'], marker='o')
plt.title('SVM Linear Kernel - Mean CV Accuracy vs C')
plt.xlabel('C')
plt.ylabel('Mean CV Accuracy')
plt.xscale('log')
plt.grid(True)
plt.savefig("../results/svm_linear_cv.png")
plt.close()

# RBF kernel
rbf_results = results_df[results_df['param_kernel'] == 'rbf']
plt.figure()
for gamma in sorted(rbf_results['param_gamma'].unique()):
    subset = rbf_results[rbf_results['param_gamma'] == gamma]
    plt.plot(subset['param_C'], subset['mean_test_score'], marker='o', label=f"gamma={gamma}")
plt.title('SVM RBF Kernel - Mean CV Accuracy vs C')
plt.xlabel('C')
plt.ylabel('Mean CV Accuracy')
plt.xscale('log')
plt.legend()
plt.grid(True)
plt.savefig("../results/svm_rbf_cv.png")
plt.close()

print("📁 Cross-validation plots saved in results folder!")
