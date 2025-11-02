# -*- coding: utf-8 -*-
"""
Machine Learning for Health - Lab 2
Parkinson's Disease UPDRS Prediction using K-NN with Local Linear Least Squares

@author: Monica Visintin

Objective: Regress Total UPDRS (Unified Parkinson's Disease Rating Scale) 
from other features using K-Nearest Neighbors with Local Linear Least Squares.

Algorithm: K-NN with Local LLS
- For each validation point, find K nearest neighbors in the training set
- Fit a local linear model using only those K neighbors
- Use the local model to predict the target value for that specific point

The UPDRS is a rating scale used to follow the progression of Parkinson's disease.
"""

# =============================================================================
# IMPORT LIBRARIES
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set pandas display options for better readability of numerical outputs
pd.set_option('display.precision', 3)

# =============================================================================
# DATA LOADING AND INITIAL EXPLORATION
# =============================================================================

# Close all previously opened figures to avoid clutter in the output
plt.close('all')

# Read the dataset from CSV file
# X is a Pandas DataFrame containing all patient data
# Shape: (Np, Nc) where Np = total samples --> patients, Nc = total columns --> features + target
X = pd.read_csv("parkinsons_updrs_av.csv")

# Extract feature names (column headers) as a list
features = list(X.columns)

# Get array of unique patient IDs to count distinct patients
# Some patients may have multiple recordings
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
# Np = number of samples/rows, Nc = number of features/columns (including target)
Np, Nc = X.shape

# =============================================================================
# EXPLORATORY DATA ANALYSIS
# =============================================================================

# Display statistical summary of each feature (mean, std, min, max, quartiles)
print("\nSTATISTICAL SUMMARY OF FEATURES:")
print(X.describe().T)  # Transpose for better readability (features as rows)

# Display data types and missing values information
print("\nDATASET INFO:")
print(X.info())

# =============================================================================
# DATA SHUFFLING
# =============================================================================
# Shuffling prevents any ordering bias in the original dataset
# This ensures random distribution of patients across train/val/test sets

# Set random seed for reproducibility
# Using the same seed will always produce the same shuffle
np.random.seed(355074)
# np.random.seed(361144)
# Create array of indices [0, 1, 2, ..., Np-1]
# Shape: (Np,)
indexsh = np.arange(Np)

# Randomly permute the indices in-place
# This shuffles the order without creating a new array
np.random.shuffle(indexsh)

# Apply shuffled indices to create shuffled dataset
Xsh = X.copy()  # Create a copy to avoid modifying original data
Xsh = Xsh.set_axis(indexsh, axis=0)  # Apply shuffled indices to rows
Xsh = Xsh.sort_index(axis=0)  # Sort by index to reset row numbers 0, 1, 2, ...

# =============================================================================
# TRAIN-VALIDATION-TEST SPLIT
# =============================================================================
# Split strategy:
# - 60% for training pool (which is further split into 40% true training + 20% validation)
# - 40% for testing
# This prevents data leakage and allows proper hyperparameter tuning

# Calculate number of samples for each split

Ntr = int(Np * 0.6)      # Total training pool: 60% of data
Ntt = int(Ntr * (2/3))   # True training set: 2/3 of training pool = 40% of total data
Nva = Ntr - Ntt          # Validation set: 1/3 of training pool = 20% of total data
Nte = Np - Ntr           # Test set: 40% of total data

print(f"\nDATA SPLIT:")
print(f"True Training samples: {Ntt} ({Ntt/Np*100:.1f}%)")
print(f"Validation samples: {Nva} ({Nva/Np*100:.1f}%)")
print(f"Test samples: {Nte} ({Nte/Np*100:.1f}%)")

# =============================================================================
# COMPUTE NORMALIZATION PARAMETERS FROM TRUE TRAINING DATA ONLY
# =============================================================================
# Important: We only use TRUE TRAINING data statistics to avoid data leakage
# Using validation or test statistics would give the model unfair advantage
# This simulates real-world scenario where we don't have access to future data

# Extract true training subset: rows [0:Ntt]
# Shape: (Ntt, Nc)
X_train = Xsh[0:Ntt]

# Compute mean for each column using only training data
# Shape: (Nc,) - one mean value per column
mm = X_train.mean()

# Compute standard deviation for each column using only training data
# Shape: (Nc,) - one std value per column
ss = X_train.std()

# Store normalization parameters for target variable separately
# These will be used to de-normalize predictions back to original scale
my = mm['total_UPDRS']  # Scalar - mean of target variable
sy = ss['total_UPDRS']  # Scalar - std of target variable

# =============================================================================
# NORMALIZE DATA AND PREPARE FEATURES/TARGET
# =============================================================================

# Normalize entire shuffled dataset using TRUE TRAINING statistics
# Formula: X_norm = (X - mean) / std
# This transforms features to have mean=0 and std=1 (z-score normalization)
# Shape: (Np, Nc)
Xsh_norm = (Xsh - mm) / ss

# Extract normalized target variable (regressand) as separate vector
# Shape: (Np,)
ysh_norm = Xsh_norm['total_UPDRS'].copy()

# Remove target and identifier columns to create feature matrix (regressors)
# 'subject#' is removed because it's an identifier, not a predictive feature
# 'total_UPDRS' is removed because it's our target variable
Xsh_norm = Xsh_norm.drop(['total_UPDRS', 'subject#'], axis=1)

# MODULAR DROP: Remove redundant or time-based features
# 'Jitter:DDP' and 'Shimmer:DDA' are derived features (redundant)
# 'test_time' is removed as it's a temporal identifier, not predictive
Xsh_norm = Xsh_norm.drop(['Jitter:DDP', 'Shimmer:DDA', 'test_time'], axis=1)
# Optionally remove 'motor_UPDRS' if it's too correlated with total_UPDRS
# Xsh_norm = Xsh_norm.drop(['motor_UPDRS'], axis=1)

# Get final list of regressor names after dropping columns
regressors = list(Xsh_norm.columns)
Nf = len(regressors)  # Number of features used for regression

print(f"\nFEATURES FOR REGRESSION:")
print(f"Number of regressors: {Nf}")
print(f"Regressors: {regressors}")

# Convert from Pandas DataFrame to NumPy arrays for numerical computation
# This is more efficient for matrix operations
Xsh_norm = Xsh_norm.values  # Shape: (Np, Nf) - Feature matrix
ysh_norm = ysh_norm.values  # Shape: (Np,) - Target vector

# Split normalized data into training, validation, and test sets
# Using the same split indices as before
X_train_norm = Xsh_norm[0:Ntt]      # Shape: (Ntt, Nf) - True training features
X_val_norm = Xsh_norm[Ntt:Ntr]      # Shape: (Nva, Nf) - Validation features
X_test_norm = Xsh_norm[Ntr:]        # Shape: (Nte, Nf) - Test features

y_train_norm = ysh_norm[0:Ntt]      # Shape: (Ntt,) - True training targets   
y_val_norm = ysh_norm[Ntt:Ntr]      # Shape: (Nva,) - Validation targets
y_test_norm = ysh_norm[Ntr:]        # Shape: (Nte,) - Test targets

print(f"\nNormalized data shapes:")
print(f"X_train_norm: {X_train_norm.shape}, X_val_norm: {X_val_norm.shape}, X_test_norm: {X_test_norm.shape}")

# =============================================================================
# K-NEAREST NEIGHBORS WITH LOCAL LINEAR LEAST SQUARES (K-NN + LLS)
# =============================================================================

print("\n" + "=" * 80)
print("K-NEAREST NEIGHBORS WITH LOCAL LINEAR LEAST SQUARES")
print("=" * 80)

# Scope of this function is to regress localli over the real values (see image of the lab with blue lines and red one)
# For each validation point, we find K nearest neighbors in the training set,
# fit a local linear model using only those K neighbors, and use that model to predict the target value for that specific point.

# The best K is the value that makes the line adapt better to the dot-like line (see image)
# the best K is the value with the lowest MSE

def knn_lls_predict(X_train, y_train, X_val, K, epsilon=1e-8):
    """
    Predict using K-Nearest Neighbors with Local Linear Least Squares.
    
    Algorithm:
    For each validation point:
    1. Compute distances to all training points
    2. Select K nearest neighbors
    3. Fit a local LLS model using only these K neighbors
    4. Use the local model to predict for that specific point
    
    This is different from global LLS which uses all training data.
    Local models adapt to the local structure of the data.
    
    Parameters:
    -----------
    X_train : ndarray, shape (Ntr, Nf)
        Training feature matrix (normalized)
        Ntr = number of training samples
        Nf = number of features
    y_train : ndarray, shape (Ntr,)
        Training target vector (normalized)
    X_val : ndarray, shape (Nval, Nf)
        Validation feature matrix (normalized)
        Nval = number of validation samples
    K : int
        Number of nearest neighbors to use for local regression
        Must satisfy: K >= Nf for matrix invertibility (without regularization)
    epsilon : float, default=1e-8
        Regularization parameter (added to diagonal of A^T A)
        Ensures matrix is always invertible even when K < Nf
        Prevents numerical instability when features are collinear
    
    Returns:
    --------
    predictions : ndarray, shape (Nval,)
        Predicted target values for validation set (normalized)
    
    Time Complexity: O(Nval * Ntr * Nf)
    Space Complexity: O(Ntr + K*Nf)
    """
    # Get dimensions
    Nval = X_val.shape[0]   # Number of validation points to predict
    Nf = X_train.shape[1]   # Number of features
    
    # Initialize prediction array with zeros
    # Shape: (Nval,)
    predictions = np.zeros(Nval)
    
    # Create identity matrix for ridge regularization
    # Shape: (Nf, Nf) - diagonal matrix with 1s on diagonal
    I = np.eye(Nf)
    
    # Loop over each validation point
    # Each point gets its own local model
    for i, x in enumerate(X_val):

        ################
        #   KNN PART   #
        ################

        # x is a single validation point
        # Shape: (Nf,) - row vector of features
        
        # STEP 1.1: Compute squared Euclidean distances to all training points
        # (Ntr, Nf) - (Nf,) = (Ntr, Nf)
        diff = X_train - x
        
        # Sum of squared differences along feature dimension
        # Shape: (Ntr,) - one distance value per training point
        # squared_distances[j] = sum((X_train[j] - x)^2) = ||X_train[j] - x||^2
        squared_distances = np.sum(diff**2, axis=1)
        
        # STEP 1.2: Find indices of K nearest neighbors
        # np.argsort returns indices that would sort the array
        # [:K] selects the first K indices (smallest distances)
        # Shape: (K,) - indices of K nearest training points
        k_indices = np.argsort(squared_distances)[:K]
        




        ################
        #   LLS PART   #
        ################

        # STEP 1.3: Extract K nearest neighbors and their targets
        # Create local feature matrix A using only K nearest neighbors
        # Shape: (K, Nf)
        A = X_train[k_indices] # Matrix X in theory notes (see LLS in lab00)
        
        # Create local target vector y using only K nearest neighbors
        # Shape: (K,)
        y_local = y_train[k_indices]
        
        # Compute local weight vector using ridge-regularized LLS
        # Normal equation with ridge regularization:
        # w_hat = (A^T A + εI)^(-1) A^T y
        
        # COURSE NOTE: 
        # w_hat = (X^T*X)^(-1)*(X^T*y)

        # ε is a small regularization parameter
        
        # Compute Gram matrix: A^T @ A
        # Shape: (Nf, Nf) - square matrix
        ATA = A.T @ A
        
        # Compute cross-correlation: A^T @ y
        # Shape: (Nf,) - vector
        ATy = A.T @ y_local

        # Add regularization term εI to ensure invertibility
        # This prevents singular matrix errors when:
        # - K < Nf (fewer neighbors than features)
        # - Features are collinear (linearly dependent)
        # Shape: (Nf, Nf)
        regularized_matrix = ATA + epsilon * I
        
        # Solve for weight vector: w_hat = (A^T A + εI)^(-1) A^T y
        # Shape: (Nf,) - local linear model weights
        w_hat = np.linalg.inv(regularized_matrix) @ ATy
        
        # STEP 1.4: Predict for this validation point using local model
        # Linear prediction: y_hat = x^T @ w_hat = sum(x[j] * w_hat[j])
        # Shape: scalar (dot product of two (Nf,) vectors)
        y_hat = x @ w_hat
        
        # Store prediction for this validation point
        predictions[i] = y_hat
    
    # Return all predictions
    # Shape: (Nval,)
    return predictions

# =============================================================================
# STEP 1: TEST WITH FIXED K VALUE (K=20)
# =============================================================================
# Before searching for optimal K, test with a reasonable fixed value
# This helps verify the implementation works correctly

print("\n" + "-" * 80)
print("STEP 1: Testing with K = 20")
print("-" * 80)

# Set fixed parameters
K_fixed = 20        # Number of neighbors (arbitrary reasonable choice)
epsilon = 1e-8      # Small regularization parameter (standard value)

# Predict on validation set using normalized data
# Input shapes: (Ntt, Nf), (Ntt,), (Nva, Nf), scalar, scalar
# Output shape: (Nva,)
y_hat_val_norm_knn_lls = knn_lls_predict(X_train_norm, y_train_norm, X_val_norm, K_fixed, epsilon)

# De-normalize predictions: y = y_norm * std + mean
# This converts predictions back to original UPDRS scale
# Shape: (Nva,)
y_hat_val_knn_lls = y_hat_val_norm_knn_lls * sy + my

# De-normalize true validation targets for comparison
# Shape: (Nva,)
y_val = y_val_norm * sy + my

# Compute prediction errors (residuals)
# e = y_true - y_predicted
# Shape: (Nva,)
E_val_knn_lls = y_val - y_hat_val_knn_lls

# Compute Mean Squared Error (MSE)
# MSE = mean(e^2) = average of squared errors
# Lower MSE indicates better predictions
# Scalar value
E_val_MSE_knn_lls = np.mean(E_val_knn_lls**2)

print(f"\nK = {K_fixed}:")
print(f"  Validation MSE: {E_val_MSE_knn_lls:.3f}")

# =============================================================================
# STEP 2-5: FIND OPTIMAL K VALUE
# =============================================================================
# Hyperparameter tuning: search for K that minimizes validation MSE
# This is model selection using the validation set


# Define search space for K (hyperparameter range)
Kmin = 60   # Minimum K to test (60 nearest neighbors)
Kmax = 300   # Maximum K to test (300 nearest neighbors)
K_step = 1  # Step size for K values (test every integer)

# Create array of K values to test
# Shape: (241,) - [60, 61, 62, ..., 300]
K_values = np.arange(Kmin, Kmax + 1, K_step)

# Initialize list to store validation MSE for each K
# Will become array of shape: (241,)
val_mse_values = []

# Loop over each K value to evaluate performance
for K in K_values:
    # Predict on validation set with current K
    # Shape: (Nva,)
    y_hat_val_norm = knn_lls_predict(X_train_norm, y_train_norm, X_val_norm, K, epsilon)
    
    # De-normalize predictions to original scale
    # Shape: (Nva,)
    y_hat_val = y_hat_val_norm * sy + my
    
    # Compute MSE for this K value
    # Scalar value
    mse = np.mean((y_val - y_hat_val)**2)
    
    # Store MSE in list
    val_mse_values.append(mse)

# Convert list to NumPy array for easier manipulation
# Shape: (200,)
val_mse_values = np.array(val_mse_values)

# Find index of minimum MSE
# Scalar - index in range [0, 199]
best_k_idx = np.argmin(val_mse_values)

# Extract optimal K and its corresponding MSE
K_opt = K_values[best_k_idx]  # Scalar - K value with lowest validation MSE
best_mse = val_mse_values[best_k_idx]  # Scalar - minimum validation MSE

# Find K with highest MSE for comparison
worst_k_idx = np.argmax(val_mse_values)  # Scalar - index of maximum MSE
K_worst = K_values[worst_k_idx]  # Scalar - K value with highest validation MSE
worst_mse = val_mse_values[worst_k_idx]  # Scalar - maximum validation MSE

print("\n" + "=" * 80)
print(f"OPTIMAL K VALUE: {K_opt}")
print(f"MINIMUM VALIDATION MSE: {best_mse:.3f}")
print("-" * 80)
print(f"WORST K VALUE: {K_worst}")
print(f"MAXIMUM VALIDATION MSE: {worst_mse:.3f}")
print("=" * 80)

# =============================================================================
# VISUALIZE MSE vs K
# =============================================================================
# Plot validation MSE as a function of K to visualize model selection

plt.figure(figsize=(12, 6))

# Plot MSE curve: line plot with circular markers
plt.plot(K_values, val_mse_values, '-o', linewidth=2, markersize=6, 
            label='Validation MSE', alpha=0.7)

# Add vertical line at optimal K for reference
plt.axvline(x=K_opt, color='r', linestyle='--', linewidth=2, 
            label=f'Optimal K = {K_opt}')

# Highlight optimal point with a star marker
plt.scatter([K_opt], [best_mse], color='red', s=200, zorder=5, marker='*', 
            edgecolors='black', linewidths=2, label=f'Min MSE = {best_mse:.3f}')

# Add axis labels and title
plt.xlabel('K (Number of Neighbors)', fontsize=12)
plt.ylabel('Validation MSE', fontsize=12)
plt.title('K-NN with Local LLS: Validation MSE vs K', fontsize=14)

# Add grid for better readability
plt.grid(alpha=0.3)

# Add legend
plt.legend(fontsize=11)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save figure to file
plt.savefig('./pictures/k_vs_mse_knn_lls.png', dpi=300)

# Display figure
plt.show()

# =============================================================================
# FINAL MODEL EVALUATION WITH OPTIMAL K
# =============================================================================
# After finding optimal K, evaluate performance on all three datasets
# This gives final assessment of model quality

print("\n" + "=" * 80)
print(f"FINAL MODEL EVALUATION WITH K_opt = {K_opt}")
print("=" * 80)

# =============================================================================
# POINT 1: TEST PHASE WITH K-NN LOCAL LLS (K = K_opt)
# =============================================================================
# Make predictions on all three sets using optimal K

# Training set predictions (to check for overfitting)
# Note: For K-NN, training predictions may show some bias
# because each point uses its K nearest neighbors (including itself if K > 1)
# Input: (Ntt, Nf), (Ntt,), (Ntt, Nf) - Output: (Ntt,)
y_hat_train_norm_final = knn_lls_predict(X_train_norm, y_train_norm, X_train_norm, K_opt, epsilon)

# Validation set predictions (used for model selection)
# Input: (Ntt, Nf), (Ntt,), (Nva, Nf) - Output: (Nva,)
y_hat_val_norm_final = knn_lls_predict(X_train_norm, y_train_norm, X_val_norm, K_opt, epsilon)

# Test set predictions (final unbiased performance estimate)
# Input: (Ntt, Nf), (Ntt,), (Nte, Nf) - Output: (Nte,)
y_hat_test_norm_final = knn_lls_predict(X_train_norm, y_train_norm, X_test_norm, K_opt, epsilon)

# De-normalize all true targets back to original UPDRS scale
y_train = y_train_norm * sy + my  # Shape: (Ntt,)
y_val = y_val_norm * sy + my      # Shape: (Nva,)
y_test = y_test_norm * sy + my    # Shape: (Nte,)

# De-normalize all predictions back to original UPDRS scale
y_hat_train_final = y_hat_train_norm_final * sy + my  # Shape: (Ntt,)
y_hat_val_final = y_hat_val_norm_final * sy + my      # Shape: (Nva,)
y_hat_test_final = y_hat_test_norm_final * sy + my    # Shape: (Nte,)

# Compute ESTIMATION errors for each set 
E_train_final = y_train - y_hat_train_final  # Shape: (Ntt,)
E_val_final = y_val - y_hat_val_final        # Shape: (Nva,)
E_test_final = y_test - y_hat_test_final     # Shape: (Nte,)

# =============================================================================
# POINT 2: DETAILED STATISTICS FOR TEST SET (K-NN LOCAL LLS)
# =============================================================================
# Compute comprehensive error statistics for the test set

# Basic error statistics
# This shows if there's systematic bias (should be close to 0)
mean_error = np.mean(E_test_final)
print(f"\nTEST SET ERROR STATISTICS (K-NN Local LLS with K={K_opt}):")
print(f"Mean Error: {mean_error:.3f}")


# This measures the spread/variability of prediction errors
std_error = np.std(E_test_final)
print(f"Standard Deviation of Errors: {std_error:.3f}")

# MSE = average of squared errors (penalizes large errors more)
mse = np.mean(E_test_final**2)
print(f"Mean Squared Error (MSE): {mse:.3f}")


# RMSE is in the same units as the target variable
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error (RMSE): {rmse:.3f}")

# Correlation and determination coefficients
corr_coef = np.corrcoef(y_test, y_hat_test_final)[0, 1]
print(f"Correlation Coefficient: {corr_coef:.3f}")


# R^2 = 1 -[MSE / np.var(y_test)]
# Measures proportion of variance explained by the model
# Range: (-∞, 1], where 1 = perfect predictions
r_squared = 1 - mse / np.var(y_test)
print(f"R² (Coefficient of Determination): {r_squared:.3f}")

# Regression line for predicted vs true values
# Fit linear regression: predicted = a * true + b
# Perfect predictions would give slope=1, intercept=0
slope, intercept = np.polyfit(y_test, y_hat_test_final, 1)
print(f"Regression Line: slope = {slope:.3f}, intercept = {intercept:.3f}")

# =============================================================================
# POINT 3: STANDARD LLS REGRESSION (GLOBAL MODEL FROM LAB 1)
# =============================================================================
# Implement standard Linear Least Squares using ALL training data
# This serves as a baseline to compare against K-NN Local LLS


#  w_hat = (X^T X)^(-1) X^T y
# Shape: (Nf,) - one weight per feature
# Use standard LLS and use dataset used for traning without the validation set splitted
X_train_norm_lls = Xsh_norm[0:Ntr]      # Shape: (Ntr, Nf) - Training features (including validation set)
y_train_norm_lls = ysh_norm[0:Ntr]      # Shape: (Ntr,) - Training targets (including validation set)

X_test_norm_lls = Xsh_norm[Ntr:]        # Shape: (Nte, Nf) - Test features
y_test_norm_lls = ysh_norm[Ntr:]        # Shape: (Nte,) - Test targets

w_hat_lls = np.linalg.inv(X_train_norm_lls.T @ X_train_norm_lls) @ (X_train_norm_lls.T @ y_train_norm_lls)

# Compute predictions on test set
# Shape: (Nte,)
y_hat_test_norm_lls = X_test_norm_lls @ w_hat_lls

# De-normalize LLS test predictions back to original scale
# Shape: (Nte,)
y_hat_test_lls = y_hat_test_norm_lls * sy + my

mean_error_lls = np.mean(y_test - y_hat_test_lls)
std_error_lls = np.std(y_test - y_hat_test_lls)
error_lls = y_test - y_hat_test_lls
mse_lls = np.mean(error_lls**2)
corr_coef_lls = np.corrcoef(y_test, y_hat_test_lls)[0, 1]
R2_lls = 1 - mse_lls / np.var(y_test)

# Compute regression line parameters for LLS
slope_lls, intercept_lls = np.polyfit(y_test, y_hat_test_lls, 1)

# =============================================================================
# COMPARISON: K-NN LOCAL LLS vs STANDARD LLS
# =============================================================================

# =============================================================================
# CREATE COMPARISON TABLE USING PANDAS DATAFRAME
# =============================================================================
comparison_data = {
    'K-NN Local LLS': [
        mean_error,
        std_error,
        mse,
        rmse,
        corr_coef,
        r_squared,
        slope,
        intercept
    ],
    'Standard LLS': [
        mean_error_lls,
        std_error_lls,
        mse_lls,
        np.sqrt(mse_lls),
        corr_coef_lls,
        R2_lls,
        slope_lls,
        intercept_lls
    ]
}

comparison_index = [
    'Mean Error',
    'Std Dev Error',
    'MSE',
    'RMSE',
    'Correlation',
    'R²',
    'Slope',
    'Intercept'
]

comparison_df = pd.DataFrame(comparison_data, index=comparison_index)
comparison_df['Difference'] = comparison_df['K-NN Local LLS'] - comparison_df['Standard LLS']

print("\n" + "=" * 80)
print("COMPARISON: K-NN LOCAL LLS vs STANDARD LLS")
print("=" * 80)
print(comparison_df)
print("=" * 80)

# =============================================================================
# VISUALIZATION: SIDE-BY-SIDE ERROR HISTOGRAMS
# =============================================================================
# Compare error distributions for both methods

# Use common bins for fair comparison
m = min(np.min(E_test_final), np.min(error_lls))
M = max(np.max(E_test_final), np.max(error_lls))
common_bins = np.linspace(m, M, 51)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# K-NN Local LLS histogram
axes[0].hist(E_test_final, bins=common_bins, density=True, alpha=0.75, 
             color='tab:blue', edgecolor='k', label='K-NN Local LLS')
axes[0].axvline(mean_error, color='r', linestyle='--', linewidth=2, 
                label=f"Mean = {mean_error:.2f}")
axes[0].set_title(f"K-NN Local LLS (K={K_opt})\nMSE={mse:.3f}, R²={r_squared:.3f}", fontsize=12)
axes[0].set_xlabel("e = y - ŷ", fontsize=11)
axes[0].set_ylabel("Probability Density", fontsize=11)
axes[0].grid(alpha=0.3)
axes[0].legend(fontsize=10)

# Standard LLS histogram
axes[1].hist(error_lls, bins=common_bins, density=True, alpha=0.75, 
             color='tab:orange', edgecolor='k', label='Standard LLS')
axes[1].axvline(mean_error_lls, color='r', linestyle='--', linewidth=2, 
                label=f"Mean = {mean_error_lls:.2f}")
axes[1].set_title(f"Standard LLS\nMSE={mse_lls:.3f}, R²={R2_lls:.3f}", fontsize=12)
axes[1].set_xlabel("e = y - ŷ", fontsize=11)
axes[1].set_ylabel("Probability Density", fontsize=11)
axes[1].grid(alpha=0.3)
axes[1].legend(fontsize=10)

plt.tight_layout()
plt.savefig('./pictures/error_histograms_comparison.png', dpi=300)

# =============================================================================
# VISUALIZATION: SIDE-BY-SIDE PREDICTION SCATTER PLOTS
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

axes[0].scatter(y_test, y_hat_test_final, alpha=0.5, s=30, 
                label='Predictions', color='tab:blue')
v = axes[0].axis()
axes[0].plot([v[0], v[1]], [v[0], v[1]], 'r-', linewidth=2, 
             label='Perfect (y=ŷ)')
y_fit = slope * y_test + intercept
axes[0].plot(y_test, y_fit, 'g--', linewidth=2, alpha=0.7,
             label=f'Fit: ŷ={slope:.2f}y+{intercept:.2f}')
axes[0].set_xlabel('True Total UPDRS (y)', fontsize=11)
axes[0].set_ylabel('Predicted Total UPDRS (ŷ)', fontsize=11)
axes[0].set_title(f'K-NN Local LLS (K={K_opt})\nR²={r_squared:.3f}, Corr={corr_coef:.3f}', 
                  fontsize=12)
axes[0].legend(fontsize=10)
axes[0].grid(alpha=0.3)
axes[0].axis('square')

axes[1].scatter(y_test, y_hat_test_lls, alpha=0.5, s=30, 
                label='Predictions', color='tab:orange')
v = axes[1].axis()
axes[1].plot([v[0], v[1]], [v[0], v[1]], 'r-', linewidth=2, 
             label='Perfect (y=ŷ)')
y_fit_lls = slope_lls * y_test + intercept_lls
axes[1].plot(y_test, y_fit_lls, 'g--', linewidth=2, alpha=0.7,
             label=f'Fit: ŷ={slope_lls:.2f}y+{intercept_lls:.2f}')
axes[1].set_xlabel('True Total UPDRS (y)', fontsize=11)
axes[1].set_ylabel('Predicted Total UPDRS (ŷ)', fontsize=11)
axes[1].set_title(f'Standard LLS\nR²={R2_lls:.3f}, Corr={corr_coef_lls:.3f}', 
                  fontsize=12)
axes[1].legend(fontsize=10)
axes[1].grid(alpha=0.3)
axes[1].axis('square')

plt.tight_layout()
plt.savefig('./pictures/predictions_comparison.png', dpi=300)

plt.show()

# =============================================================================
# KEY INSIGHTS SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("KEY INSIGHTS:")
print("=" * 80)

# Calculate improvement metrics
mse_improvement = ((mse_lls - mse) / mse_lls) * 100

if mse < mse_lls:
    print(f"✓ K-NN Local LLS outperforms Standard LLS")
    print(f"  MSE Reduction: {mse_improvement:.2f}%")
else:
    print(f"✗ Standard LLS outperforms K-NN Local LLS")
    print(f"  MSE Increase: {-mse_improvement:.2f}%")

print(f"\nK-NN Local LLS: K={K_opt}, R²={r_squared:.3f}")
print(f"Standard LLS:   R²={R2_lls:.3f}")

print("\n" + "=" * 80)

plt.show()

# =============================================================================
# POINT 5 (EXAM): DETAILED STATISTICS FOR TRAINING SET (K-NN LOCAL LLS)
# =============================================================================
# This allows us to check for overfitting by comparing train error to test error

print("\n" + "=" * 80)
print(f"PERFORMANCE METRICS FOR TRAINING DATASET (K-NN LLS with K={K_opt})")
print("=" * 80)

# Basic error statistics
mean_error_train = np.mean(E_train_final)
print(f"Mean Error: {mean_error_train:.3f}")

std_error_train = np.std(E_train_final)
print(f"Standard Deviation of Errors: {std_error_train:.3f}")

mse_train = np.mean(E_train_final**2)
print(f"Mean Squared Error (MSE): {mse_train:.3f}")

rmse_train = np.sqrt(mse_train)
print(f"Root Mean Squared Error (RMSE): {rmse_train:.3f}")

# Correlation and determination coefficients
corr_coef_train = np.corrcoef(y_train, y_hat_train_final)[0, 1]
print(f"Correlation Coefficient: {corr_coef_train:.3f}")

# R^2 = 1 - [MSE / np.var(y_train)]
r_squared_train = 1 - mse_train / np.var(y_train)
print(f"R² (Coefficient of Determination): {r_squared_train:.3f}")

# Regression line for predicted vs true values
slope_train, intercept_train = np.polyfit(y_train, y_hat_train_final, 1)
print(f"Regression Line: slope = {slope_train:.3f}, intercept = {intercept_train:.3f}")
print("=" * 80)