# -*- coding: utf-8 -*-
"""
Machine Learning for Health - Lab 2
Parkinson's Disease UPDRS Prediction using Linear Least Squares Regression

@author: Monica Visintin

Objective: Regress Total UPDRS (Unified Parkinson's Disease Rating Scale) 
from other features in the "parkinsons_updrs_av.csv" dataset.

Algorithms implemented:
1. Linear Least Squares (LLS) - Closed-form solution

The UPDRS is a rating scale used to follow the progression of Parkinson's disease.
"""

# =============================================================================
# IMPORT LIBRARIES
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set pandas display options for better readability
pd.set_option('display.precision', 3)

# =============================================================================
# DATA LOADING AND INITIAL EXPLORATION
# =============================================================================

# Close all previously opened figures to avoid clutter
plt.close('all')

# Read the dataset from CSV file
# X is a Pandas DataFrame containing all patient data
X = pd.read_csv("parkinsons_updrs_av.csv")

# Extract feature names (column headers)
features = list(X.columns)

# Get unique patient IDs to count distinct patients
subj = pd.unique(X['subject#'])

# Display basic dataset information
print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)
print(f"Original dataset shape: {X.shape}")
print(f"Number of rows (samples): {X.shape[0]}")
print(f"Number of columns (features): {X.shape[1]}")
print(f"Number of distinct patients: {len(subj)}")
print(f"\nFeature names ({len(features)} total):")
for i, feature in enumerate(features, 1):
    print(f"  {i}. {feature}")
print("=" * 60)

# Store dataset dimensions
Np, Nc = X.shape  # Np = number of rows/samples, Nc = number of features (including target)

# =============================================================================
# EXPLORATORY DATA ANALYSIS
# =============================================================================

# Display statistical summary of each feature
print("\nSTATISTICAL SUMMARY OF FEATURES:")
print(X.describe().T)  # Transpose for better readability

# Display data types and missing values information
print("\nDATASET INFO:")
print(X.info())

# =============================================================================
# DATA SHUFFLING
# =============================================================================
# Shuffling prevents any ordering bias in the original dataset

# Set random seed for reproducibility
np.random.seed(355074)

# Create index array and shuffle it
indexsh = np.arange(Np)  # Generate array [0, 1, ..., Np-1]
np.random.shuffle(indexsh)  # Randomly permute the indices

# Apply shuffled indices to create shuffled dataset
Xsh = X.copy()  # Create a copy to avoid modifying original data
Xsh = Xsh.set_axis(indexsh, axis=0)  # Apply shuffled indices
Xsh = Xsh.sort_index(axis=0)  # Sort by index to reset row numbers

# =============================================================================
# TRAIN-TEST SPLIT
# =============================================================================

# Split dataset: 60% training-->40% True Training Dataset and 20% validation , 40% testing
Ntr = int(Np * 0.6)  # Number of training samples
Ntt = int(Ntr * (2/3))  # Number of true training samples (2/3 of training set)
Nva = Ntr - Ntt   # Number of validation samples (1/3 of training set)
Nte = Np - Ntr       # Number of test samples

print(f"\nDATA SPLIT:")
print(f"True Training samples: {Ntt} ({Ntt/Np*100:.1f}%)")
print(f"Validation samples: {Nva} ({Nva/Np*100:.1f}%)")
print(f"Test samples: {Nte} ({Nte/Np*100:.1f}%)")

# ==========================================================================================
# COMPUTE NORMALIZATION PARAMETERS FROM TRUE TRAINING DATA ONLY (DATA PREPARATION - POINT 4)
# ==========================================================================================
# Important: We only use training data statistics to avoid data leakage

# Extract training subset
X_train = Xsh[0:Ntt]

# Compute mean and standard deviation for each feature (training data only)
mm = X_train.mean()  # Mean of each column
ss = X_train.std()   # Standard deviation of each column

# Store target variable (total_UPDRS) statistics separately
my = mm['total_UPDRS']  # Mean of target variable
sy = ss['total_UPDRS']  # Standard deviation of target variable

# =============================================================================
# NORMALIZE DATA AND PREPARE FEATURES/TARGET
# =============================================================================

# Normalize entire shuffled dataset using TRUE training statistics
Xsh_norm = (Xsh - mm) / ss

# Extract normalized target variable (regressand)
ysh_norm = Xsh_norm['total_UPDRS'].copy()

# Remove target and subject ID from features (regressors)
# subject# is removed because it's an identifier, not a predictive feature
Xsh_norm = Xsh_norm.drop(['total_UPDRS', 'subject#'], axis=1)


# MODULAR DROP
Xsh_norm = Xsh_norm.drop(['Jitter:DDP', 'Shimmer:DDA', 'test_time'], axis=1)
# Xsh_norm = Xsh_norm.drop(['motor_UPDRS'], axis=1)


# Get final list of regressor names
regressors = list(Xsh_norm.columns)
Nf = len(regressors)  # Number of features used for regression

print(f"\nFEATURES FOR REGRESSION:")
print(f"Number of regressors: {Nf}")
print(f"Regressors: {regressors}")

# Convert from DataFrame to NumPy arrays for numerical computation
Xsh_norm = Xsh_norm.values  # Feature matrix
ysh_norm = ysh_norm.values  # Target vector

# Split normalized data into training, validation and test sets
X_train_norm = Xsh_norm[0:Ntt]      # True Training features
X_val_norm = Xsh_norm[Ntt:Ntr]    # Validation features
X_test_norm = Xsh_norm[Ntr:]       # Test features

y_train_norm = ysh_norm[0:Ntt]      # True Training target   
y_val_norm = ysh_norm[Ntt:Ntr]    # Validation target
y_test_norm = ysh_norm[Ntr:]       # Test target


print(f"\nNormalized data shapes:")
print(f"X_train_norm: {X_train_norm.shape}, X_val_norm: {X_val_norm.shape}, X_test_norm: {X_test_norm.shape}")

# =============================================================================
# ALGORITHM: LINEAR LEAST SQUARES (LLS) REGRESSION
# =============================================================================
# Solve the normal equations: (X^T X) w = X^T y
# Solution: w_hat = (X^T X)^(-1) X^T y
# This is a closed-form solution (direct computation)

print("\n" + "=" * 80)
print("LINEAR LEAST SQUARES (LLS) REGRESSION")
print("=" * 80)

# Compute optimal weight vector using closed-form solution on TRUE TRAINING data
w_hat_lls = np.linalg.inv(X_train_norm.T @ X_train_norm) @ (X_train_norm.T @ y_train_norm)

# Make predictions on all sets (normalized)
y_hat_train_norm_lls = X_train_norm @ w_hat_lls  # True training predictions
y_hat_val_norm_lls = X_val_norm @ w_hat_lls  # Validation predictions
y_hat_test_norm_lls = X_test_norm @ w_hat_lls  # Test predictions

# De-normalize predictions and targets (convert back to original scale)
y_train = y_train_norm * sy + my                    # True training values
y_val = y_val_norm * sy + my                        # Validation values
y_test = y_test_norm * sy + my                      # Test values
y_hat_train_lls = y_hat_train_norm_lls * sy + my   # Predicted true training values
y_hat_val_lls = y_hat_val_norm_lls * sy + my       # Predicted validation values
y_hat_test_lls = y_hat_test_norm_lls * sy + my     # Predicted test values

# Compute errors
E_train_lls = y_train - y_hat_train_lls  # True training errors
E_val_lls = y_val - y_hat_val_lls        # Validation errors
E_test_lls = y_test - y_hat_test_lls     # Test errors

# Compute performance metrics
E_train_MSE_lls = np.mean(E_train_lls**2)
R2_train_lls = 1 - E_train_MSE_lls / np.var(y_train)
E_val_MSE_lls = np.mean(E_val_lls**2)
R2_val_lls = 1 - E_val_MSE_lls / np.var(y_val)
E_test_MSE_lls = np.mean(E_test_lls**2)
R2_test_lls = 1 - E_test_MSE_lls / np.var(y_test)

print(f"True Training MSE: {E_train_MSE_lls:.3f}, R²: {R2_train_lls:.3f}")
print(f"Validation MSE: {E_val_MSE_lls:.3f}, R²: {R2_val_lls:.3f}")
print(f"Test MSE: {E_test_MSE_lls:.3f}, R²: {R2_test_lls:.3f}")

# =============================================================================
# VISUALIZE LEARNED WEIGHTS
# =============================================================================

nn = np.arange(Nf)  # Feature indices

plt.figure(figsize=(12, 6))
plt.plot(nn, w_hat_lls, '-o', linewidth=2, markersize=8, label='LLS (closed-form)', alpha=0.7)
plt.xticks(nn, regressors, rotation=90)
plt.ylabel(r'$\hat{w}(n)$', fontsize=12)
plt.xlabel('Feature Index', fontsize=11)
plt.title('Learned Weights - Linear Least Squares', fontsize=14)
plt.legend(fontsize=11)
plt.grid(alpha=0.3)
plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('./pictures/weights_lls.png', dpi=300)
plt.draw()

# =============================================================================
# VISUALIZE ERROR DISTRIBUTIONS
# =============================================================================

# Determine common bin edges
M = np.max([np.max(E_train_lls), np.max(E_val_lls), np.max(E_test_lls)])
m = np.min([np.min(E_train_lls), np.min(E_val_lls), np.min(E_test_lls)])
common_bins = np.linspace(m, M, 51)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# True training errors
axes[0].hist(E_train_lls, bins=common_bins, density=True, alpha=0.7, color='blue')
axes[0].set_xlabel(r'$e = y - \hat{y}$ (Prediction Error)', fontsize=11)
axes[0].set_ylabel('Probability Density', fontsize=11)
axes[0].set_title('True Training Error Distribution', fontsize=12)
axes[0].grid(alpha=0.3)

# Validation errors
axes[1].hist(E_val_lls, bins=common_bins, density=True, alpha=0.7, color='green')
axes[1].set_xlabel(r'$e = y - \hat{y}$ (Prediction Error)', fontsize=11)
axes[1].set_ylabel('Probability Density', fontsize=11)
axes[1].set_title('Validation Error Distribution', fontsize=12)
axes[1].grid(alpha=0.3)

# Test errors
axes[2].hist(E_test_lls, bins=common_bins, density=True, alpha=0.7, color='orange')
axes[2].set_xlabel(r'$e = y - \hat{y}$ (Prediction Error)', fontsize=11)
axes[2].set_ylabel('Probability Density', fontsize=11)
axes[2].set_title('Test Error Distribution', fontsize=12)
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('./pictures/error_distribution.png', dpi=300)
plt.draw()

# =============================================================================
# VISUALIZE PREDICTIONS: TRUE vs PREDICTED
# =============================================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# True training predictions
axes[0].plot(y_train, y_hat_train_lls, 'o', alpha=0.5, label='LLS predictions')
v = axes[0].axis()
axes[0].plot([v[0], v[1]], [v[0], v[1]], 'r-', linewidth=2, label='Perfect prediction')
axes[0].set_xlabel(r'True Total UPDRS ($y$)', fontsize=11)
axes[0].set_ylabel(r'Predicted Total UPDRS ($\hat{y}$)', fontsize=11)
axes[0].set_title(f'True Training Set (R²={R2_train_lls:.3f})', fontsize=12)
axes[0].legend(fontsize=10)
axes[0].grid(alpha=0.3)
axes[0].axis('square')

# Validation predictions
axes[1].plot(y_val, y_hat_val_lls, 'o', alpha=0.5, label='LLS predictions', color='green')
v = axes[1].axis()
axes[1].plot([v[0], v[1]], [v[0], v[1]], 'r-', linewidth=2, label='Perfect prediction')
axes[1].set_xlabel(r'True Total UPDRS ($y$)', fontsize=11)
axes[1].set_ylabel(r'Predicted Total UPDRS ($\hat{y}$)', fontsize=11)
axes[1].set_title(f'Validation Set (R²={R2_val_lls:.3f})', fontsize=12)
axes[1].legend(fontsize=10)
axes[1].grid(alpha=0.3)
axes[1].axis('square')

# Test predictions
axes[2].plot(y_test, y_hat_test_lls, 'o', alpha=0.5, label='LLS predictions', color='orange')
v = axes[2].axis()
axes[2].plot([v[0], v[1]], [v[0], v[1]], 'r-', linewidth=2, label='Perfect prediction')
axes[2].set_xlabel(r'True Total UPDRS ($y$)', fontsize=11)
axes[2].set_ylabel(r'Predicted Total UPDRS ($\hat{y}$)', fontsize=11)
axes[2].set_title(f'Test Set (R²={R2_test_lls:.3f})', fontsize=12)
axes[2].legend(fontsize=10)
axes[2].grid(alpha=0.3)
axes[2].axis('square')

plt.tight_layout()
plt.savefig('./pictures/predictions_lls.png', dpi=300)
plt.draw()

# =============================================================================
# PERFORMANCE SUMMARY TABLE
# =============================================================================

cols = ['MSE', 'R²']
rows = ['True Training', 'Validation', 'Test']
performance = pd.DataFrame([
    [E_train_MSE_lls, R2_train_lls],
    [E_val_MSE_lls, R2_val_lls],
    [E_test_MSE_lls, R2_test_lls]
], columns=cols, index=rows)

print("\n" + "=" * 80)
print("PERFORMANCE SUMMARY - LINEAR LEAST SQUARES")
print("=" * 80)
print(performance)
print("=" * 80)

# Display all figures
plt.show()