# -*- coding: utf-8 -*-
"""
Machine Learning for Health - Lab 1
Parkinson's Disease UPDRS Prediction using Different Regression Algorithms

@author: Monica Visintin

Objective: Regress Total UPDRS (Unified Parkinson's Disease Rating Scale) 
from other features in the "parkinsons_updrs_av.csv" dataset.

Algorithms implemented:
1. Linear Least Squares (LLS) - Closed-form solution
2. Steepest Descent (Gradient Descent) - Iterative optimization

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
# CORRELATION ANALYSIS
# =============================================================================

# Normalize/standardize the data (zero mean, unit variance)
# This is necessary for meaningful correlation analysis
Xnorm = (X - X.mean()) / X.std()

# Compute covariance matrix (which equals correlation for standardized data)
c = Xnorm.cov()

# Visualize the correlation matrix as a heatmap
plt.figure(figsize=(10, 8))
plt.matshow(np.abs(c.values), fignum=0)  # Use absolute values for better visualization
plt.xticks(np.arange(len(features)), features, rotation=90)
plt.yticks(np.arange(len(features)), features, rotation=0)
plt.colorbar(label='Absolute Correlation Coefficient')
plt.title('Correlation Matrix of All Features', pad=20)
plt.tight_layout()
plt.savefig('./pictures/corr_coeff.png', dpi=300)
plt.draw()

# Plot correlation of each feature with total_UPDRS (our target variable)
plt.figure(figsize=(10, 6))
c.total_UPDRS.plot(kind='bar')
plt.grid(axis='y', alpha=0.3)
plt.xticks(np.arange(len(features)), features, rotation=90)
plt.ylabel('Correlation Coefficient')
plt.title('Correlation between total_UPDRS (target) and Other Features')
plt.tight_layout()
plt.savefig('./pictures/UPDRS_corr_coeff.png', dpi=300)
plt.draw()

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

# Alternative method (commented out):
# Xsh = X.sample(frac=1, replace=False, random_state=30, axis=0, ignore_index=True)

# =============================================================================
# TRAIN-TEST SPLIT
# =============================================================================

# Split dataset: 50% training, 50% testing
Ntr = int(Np * 0.5)  # Number of training samples
Nte = Np - Ntr       # Number of test samples

print(f"\nDATA SPLIT:")
print(f"Training samples: {Ntr} ({Ntr/Np*100:.1f}%)")
print(f"Test samples: {Nte} ({Nte/Np*100:.1f}%)")

# =============================================================================
# COMPUTE NORMALIZATION PARAMETERS FROM TRAINING DATA ONLY
# =============================================================================
# Important: We only use training data statistics to avoid data leakage

# Extract training subset
X_tr = Xsh[0:Ntr]

# Compute mean and standard deviation for each feature (training data only)
mm = X_tr.mean()  # Mean of each column
ss = X_tr.std()   # Standard deviation of each column

# Store target variable (total_UPDRS) statistics separately
my = mm['total_UPDRS']  # Mean of target variable
sy = ss['total_UPDRS']  # Standard deviation of target variable

# =============================================================================
# NORMALIZE DATA AND PREPARE FEATURES/TARGET
# =============================================================================

# Normalize entire shuffled dataset using training statistics
Xsh_norm = (Xsh - mm) / ss

# Extract normalized target variable (regressand)
ysh_norm = Xsh_norm['total_UPDRS'].copy()

# Remove target and subject ID from features (regressors)
# subject# is removed because it's an identifier, not a predictive feature
Xsh_norm = Xsh_norm.drop(['total_UPDRS', 'subject#'], axis=1)


# MODULAR DROP
Xsh_norm = Xsh_norm.drop(['Jitter:DDP', 'Shimmer:DDA'], axis=1)
Xsh_norm = Xsh_norm.drop(['motor_UPDRS'], axis=1)


# Get final list of regressor names
regressors = list(Xsh_norm.columns)
Nf = len(regressors)  # Number of features used for regression

print(f"\nFEATURES FOR REGRESSION:")
print(f"Number of regressors: {Nf}")
print(f"Regressors: {regressors}")

# Convert from DataFrame to NumPy arrays for numerical computation
Xsh_norm = Xsh_norm.values  # Feature matrix
ysh_norm = ysh_norm.values  # Target vector

# Split normalized data into training and test sets
X_tr_norm = Xsh_norm[0:Ntr]      # Training features
X_te_norm = Xsh_norm[Ntr:]       # Test features
y_tr_norm = ysh_norm[0:Ntr]      # Training target
y_te_norm = ysh_norm[Ntr:]       # Test target

print(f"\nNormalized data shapes:")
print(f"X_tr_norm: {X_tr_norm.shape}, X_te_norm: {X_te_norm.shape}")

# =============================================================================
# ALGORITHM 1: LINEAR LEAST SQUARES (LLS) REGRESSION
# =============================================================================
# Solve the normal equations: (X^T X) w = X^T y
# Solution: w_hat = (X^T X)^(-1) X^T y
# This is a closed-form solution (direct computation)

print("\n" + "=" * 80)
print("ALGORITHM 1: LINEAR LEAST SQUARES (LLS)")
print("=" * 80)

# Compute optimal weight vector using closed-form solution
w_hat_lls = np.linalg.inv(X_tr_norm.T @ X_tr_norm) @ (X_tr_norm.T @ y_tr_norm)

# Make predictions on training and test sets (normalized)
y_hat_tr_norm_lls = X_tr_norm @ w_hat_lls  # Training predictions
y_hat_te_norm_lls = X_te_norm @ w_hat_lls  # Test predictions

# De-normalize predictions and targets (convert back to original scale)
y_tr = y_tr_norm * sy + my                    # True training values
y_te = y_te_norm * sy + my                    # True test values
y_hat_tr_lls = y_hat_tr_norm_lls * sy + my   # Predicted training values
y_hat_te_lls = y_hat_te_norm_lls * sy + my   # Predicted test values

# Compute errors
E_tr_lls = y_tr - y_hat_tr_lls  # Training errors
E_te_lls = y_te - y_hat_te_lls  # Test errors

# Compute performance metrics
E_tr_MSE_lls = np.mean(E_tr_lls**2)
R2_tr_lls = 1 - E_tr_MSE_lls / np.var(y_tr)
E_te_MSE_lls = np.mean(E_te_lls**2)
R2_te_lls = 1 - E_te_MSE_lls / np.var(y_te)

print(f"Training MSE: {E_tr_MSE_lls:.3f}, R²: {R2_tr_lls:.3f}")
print(f"Test MSE: {E_te_MSE_lls:.3f}, R²: {R2_te_lls:.3f}")

# =============================================================================
# ALGORITHM 2: STEEPEST DESCENT (GRADIENT DESCENT)
# =============================================================================
# Iterative optimization algorithm
# Update rule: w(k+1) = w(k) - μ * ∇J(w(k))
# where J(w) = (1/N) * ||y - Xw||² is the Mean Squared Error cost function
# and ∇J(w) = (2/N) * X^T(Xw - y) is the gradient

print("\n" + "=" * 80)
print("ALGORITHM 2: STEEPEST DESCENT (GRADIENT DESCENT)")
print("=" * 80)

# Maximum number of iterations to prevent infinite loops
max_iterations = 10000  # Adjust based on convergence behavior

# # Convergence threshold: stop when gradient norm is below this value
# tolerance = 1e-6  # TODO: Adjust for faster/slower convergence

# INITIALIZE WEIGHTS AND HISTORIES
w_hat_sd = np.random.randn(Nf) * 0.01  # Small random values
# Arrays to store training history (for visualization)
cost_history = []      # MSE at each iteration
w_history = []         # Weight vector at each iteration
gradient_norm_history = []  # Gradient magnitude at each iteration

# IMPLEMENT GRADIENT DESCENT LOOP
print("Starting gradient descent optimization...")
for iteration in range(max_iterations):
    
    # STEP 1: evaluate the Hessian Matrix H(w_hat(iteration)) = 2/N * X^T*X
    H = (2 / Ntr) * (X_tr_norm.T @ X_tr_norm)

    # STEP 2: Start from an initial guess for the weight vector w_hat(0) 
    # w_hat_sd is already initialized before the loop

    # STEP 3: compute the gradient of the mean square error function at current weights
    # grad(e_msq) = 1/N * [(-2*X^T*y) + (2*X^T*X*w_hat_sd(iteration)) ]
    gradient = (1 / Ntr) * (-2 * X_tr_norm.T @ y_tr_norm + 2 * X_tr_norm.T @ X_tr_norm @ w_hat_sd)

    # STEP 4: update the weight vector using the steepest descent rule
    # w_hat_sd(iteration+1) = w_hat_sd(iteration) - alpha * grad(e_msq)
    # alpha = (norm_squared(grad(e_msq) / (grad(e_msq)^T * H * grad(e_msq))
    norm_squared = np.linalg.norm(gradient)**2 # numerator
    alpha = norm_squared / (gradient.T @ H @ gradient)  # step size
    w_hat_sd = w_hat_sd - alpha * gradient

    # STEP 5: Set i += 1 and repeat from STEP 3 until convergence

    # Compute current cost (MSE)
    residuals = y_tr_norm - X_tr_norm @ w_hat_sd
    cost = (1 / Ntr) * np.linalg.norm(residuals)**2
    
    # Store history for visualization
    cost_history.append(cost)
    w_history.append(w_hat_sd.copy())
    gradient_norm = np.linalg.norm(gradient)
    gradient_norm_history.append(gradient_norm)
    

print(f"Final cost: {cost_history[-1]:.6f}")
print(f"Total iterations: {len(cost_history)}")

# Make predictions with Steepest Descent weights
y_hat_tr_norm_sd = X_tr_norm @ w_hat_sd
y_hat_te_norm_sd = X_te_norm @ w_hat_sd

# De-normalize predictions
y_hat_tr_sd = y_hat_tr_norm_sd * sy + my
y_hat_te_sd = y_hat_te_norm_sd * sy + my

# Compute errors and metrics
E_tr_sd = y_tr - y_hat_tr_sd
E_te_sd = y_te - y_hat_te_sd
E_tr_MSE_sd = np.mean(E_tr_sd**2)
R2_tr_sd = 1 - E_tr_MSE_sd / np.var(y_tr)
E_te_MSE_sd = np.mean(E_te_sd**2)
R2_te_sd = 1 - E_te_MSE_sd / np.var(y_te)

print(f"Training MSE: {E_tr_MSE_sd:.3f}, R²: {R2_tr_sd:.3f}")
print(f"Test MSE: {E_te_MSE_sd:.3f}, R²: {R2_te_sd:.3f}")

# =============================================================================
# VISUALIZE STEEPEST DESCENT CONVERGENCE
# =============================================================================

# Plot 1: Cost function vs iterations
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(cost_history, linewidth=2)
plt.xlabel('Iteration', fontsize=11)
plt.ylabel('Cost (MSE)', fontsize=11)
plt.title('Cost Function Convergence', fontsize=12)
plt.grid(alpha=0.3)
plt.yscale('log')  # Log scale often shows convergence better

plt.tight_layout()
plt.savefig('./pictures/SD_convergence.png', dpi=300)
plt.draw()

# =============================================================================
# COMPARE LEARNED WEIGHTS: LLS vs STEEPEST DESCENT
# =============================================================================

nn = np.arange(Nf)  # Feature indices

plt.figure(figsize=(12, 6))
plt.plot(nn, w_hat_lls, '-o', linewidth=2, markersize=8, label='LLS (closed-form)', alpha=0.7)
plt.plot(nn, w_hat_sd, '-s', linewidth=2, markersize=6, label='Steepest Descent', alpha=0.7)
plt.xticks(nn, regressors, rotation=90)
plt.ylabel(r'$\hat{w}(n)$', fontsize=12)
plt.xlabel('Feature Index', fontsize=11)
plt.title('Comparison of Learned Weights', fontsize=14)
plt.legend(fontsize=11)
plt.grid(alpha=0.3)
plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('./pictures/weights_comparison.png', dpi=300)
plt.draw()

# =============================================================================
# COMPARE ERROR DISTRIBUTIONS
# =============================================================================

# Determine common bin edges for both algorithms
M = np.max([np.max(E_tr_lls), np.max(E_tr_sd), np.max(E_te_lls), np.max(E_te_sd)])
m = np.min([np.min(E_tr_lls), np.min(E_tr_sd), np.min(E_te_lls), np.min(E_te_sd)])
common_bins = np.linspace(m, M, 51)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Training errors
axes[0].hist([E_tr_lls, E_tr_sd], bins=common_bins, density=True, 
             histtype='bar', label=['LLS', 'Steepest Descent'], alpha=0.7)
axes[0].set_xlabel(r'$e = y - \hat{y}$ (Prediction Error)', fontsize=11)
axes[0].set_ylabel('Probability Density', fontsize=11)
axes[0].set_title('Training Error Distribution', fontsize=12)
axes[0].legend(fontsize=10)
axes[0].grid(alpha=0.3)

# Test errors
axes[1].hist([E_te_lls, E_te_sd], bins=common_bins, density=True, 
             histtype='bar', label=['LLS', 'Steepest Descent'], alpha=0.7)
axes[1].set_xlabel(r'$e = y - \hat{y}$ (Prediction Error)', fontsize=11)
axes[1].set_ylabel('Probability Density', fontsize=11)
axes[1].set_title('Test Error Distribution', fontsize=12)
axes[1].legend(fontsize=10)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('./pictures/error_comparison.png', dpi=300)
plt.draw()

# =============================================================================
# COMPARE PREDICTIONS: TRUE vs PREDICTED
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# LLS predictions
axes[0].plot(y_te, y_hat_te_lls, 'o', alpha=0.5, label='LLS predictions')
v = axes[0].axis()
axes[0].plot([v[0], v[1]], [v[0], v[1]], 'r-', linewidth=2, label='Perfect prediction')
axes[0].set_xlabel(r'True Total UPDRS ($y$)', fontsize=11)
axes[0].set_ylabel(r'Predicted Total UPDRS ($\hat{y}$)', fontsize=11)
axes[0].set_title(f'LLS - Test Set (R²={R2_te_lls:.3f})', fontsize=12)
axes[0].legend(fontsize=10)
axes[0].grid(alpha=0.3)
axes[0].axis('square')

# Steepest Descent predictions
axes[1].plot(y_te, y_hat_te_sd, 'o', alpha=0.5, label='SD predictions', color='orange')
v = axes[1].axis()
axes[1].plot([v[0], v[1]], [v[0], v[1]], 'r-', linewidth=2, label='Perfect prediction')
axes[1].set_xlabel(r'True Total UPDRS ($y$)', fontsize=11)
axes[1].set_ylabel(r'Predicted Total UPDRS ($\hat{y}$)', fontsize=11)
axes[1].set_title(f'Steepest Descent - Test Set (R²={R2_te_sd:.3f})', fontsize=12)
axes[1].legend(fontsize=10)
axes[1].grid(alpha=0.3)
axes[1].axis('square')

plt.tight_layout()
plt.savefig('./pictures/predictions_comparison.png', dpi=300)
plt.draw()

# =============================================================================
# PERFORMANCE COMPARISON TABLE
# =============================================================================

cols = ['Train_MSE', 'Train_R2', 'Test_MSE', 'Test_R2']
rows = ['LLS', 'Steepest Descent']
comparison = pd.DataFrame([
    [E_tr_MSE_lls, R2_tr_lls, E_te_MSE_lls, R2_te_lls],
    [E_tr_MSE_sd, R2_tr_sd, E_te_MSE_sd, R2_te_sd]
], columns=cols, index=rows)

print("\n" + "=" * 80)
print("ALGORITHM COMPARISON")
print("=" * 80)
print(comparison)
print("=" * 80)

# Compare weight differences
weight_difference = np.linalg.norm(w_hat_lls - w_hat_sd)
print(f"\nL2 norm of weight difference (||w_LLS - w_SD||): {weight_difference:.6f}")
print("\nInterpretation:")
print(f"- If algorithms converged correctly, weights should be very similar")
print(f"- Small differences are normal due to numerical precision and convergence criteria")
print(f"- Large differences suggest: incorrect implementation, insufficient iterations,")
print(f"  or inappropriate learning rate")
print("=" * 80)

# Display all figures
plt.show()