import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import norm
import math
import matplotlib.pyplot as plt
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Setting a seed for replication and consistency
seed = 4444
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    
# Load the choice data from Excel
path = "-"
sheet = "-"
df_total = pd.read_excel(path, sheet)
no_alts = 3

# Define attributes, here attribute names were enteres, while the labels in dataset were numbered as follows No_alt followed by attr name followed by level number; alt_1: 1Rent0, 1Rent1, 1Rent2. Alt_2: 2Rent0, 2Rent1, 2Rent2 etc. (the constant naturally doesnt have level number)
attributes = ["-", "-", "-"]
feature_cols = []
for alt in range(1, no_alts+1):
    feature_cols.append(f"{alt}Con")
    for attr in attributes:
        for level in [1, 2]:
            feature_cols.append(f"{alt}{attr}_{level}")
print("MNL feature columns:", feature_cols)

# Extract attribute data
X_raw = df_total[feature_cols].values # shape: (N, no_alts * n_vars) 
num_sessions = X_raw.shape[0] # number of choice tasks based on dataset shape (wide)
n_vars = len(feature_cols) // no_alts 

# Reshape into (N, no_alts, n_vars)
X_np = X_raw.reshape(num_sessions, no_alts, n_vars).astype(np.float32)

# Standardize columns across all alternatives 
for j in range(0, n_vars):
    col = X_np[:, :, j]
    mean_val = np.mean(col)
    std_val = np.std(col)
    if std_val != 0:
        X_np[:, :, j] = (col - mean_val) / std_val

X = torch.tensor(X_np, dtype=torch.float32)
print("X shape:", X.shape)  # shape (N, no_alts, no_vars)

# Extract observed choices (in column "choice_" coded as 1,2,3) and convert to 0-index
y = torch.tensor(df_total["choice_"].values, dtype=torch.long) - 1
print("y shape:", y.shape)  # shape is (N,)


# Load learning data 
# They are named similar to the constant: 1Age, 1Education, 2Age, 2Education etc.
learn_vars = ["-", "-", "-"]
q_cols = []
for alt in range(1, no_alts+1):
    for var in learn_vars:
        q_cols.append(f"{alt}{var}")
print("Learning part feature columns:", q_cols)

Q_raw = df_total[q_cols].values  # shape: (N, no_alts * n_q)
n_q = len(learn_vars)  
Q_np = Q_raw.reshape(num_sessions, no_alts, n_q).astype(np.float32)

# Standardize each column of learning data (across all alternatives together).
for j in range(n_q):
    col = Q_np[:, :, j]
    mean_val = np.mean(col)
    std_val = np.std(col)
    if std_val != 0:
        Q_np[:, :, j] = (col - mean_val) / std_val
Q = torch.tensor(Q_np, dtype=torch.float32)
print("Q shape:", Q.shape)  # (N, no_alts, no_vars)


# Define the nn-embedded model
class HybridModel(nn.Module):
    def __init__(self, n_vars, no_alts, n_q, hidden_size=10):
        super(HybridModel, self).__init__()
        # Knowledge-driven part
        self.theta = nn.Parameter(torch.zeros(n_vars, 1))
        # Data-driven part
        self.dnn = nn.Sequential(
            nn.Linear(n_q, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 1)
        )
    def forward(self, X, Q):
        # Knowledge-driven utility
        U_MNL = torch.matmul(X, self.theta).squeeze(2)
        # Data-driven utility
        Q_flat = Q.reshape(-1, n_q)
        r_flat = self.dnn(Q_flat) 
        r = r_flat.reshape(-1, no_alts)  
        # Total utility is the sum of knowledge and data utilities
        U_total = U_MNL + r
        return U_total, U_MNL, r

model = HybridModel(n_vars, no_alts, n_q, hidden_size=10)

loss_fn = nn.NLLLoss(reduction='sum')
# Set optimizer to Adam
optimizer = optim.Adam(model.parameters(), lr=0.15) 

# Train the model with decoupled losses, this simply means no non-linear gradient flows can go from the dnn to the mnl part, keeping the mnl linear and interpretable
def closure():
    optimizer.zero_grad()
    U_total, U_MNL, r = model(X, Q)
    # Compute log-softmax for two different utility combinations
    log_probs_theta = torch.log_softmax(U_MNL + r.detach(), dim=1)
    log_probs_dnn   = torch.log_softmax(U_MNL.detach() + r, dim=1)
    loss_theta = loss_fn(log_probs_theta, y)
    loss_dnn = loss_fn(log_probs_dnn, y)
    loss = loss_theta + loss_dnn
    loss.backward()
    return loss

losses=[]
epochs=75
for epoch in range(epochs):
    optimizer.zero_grad()  # clear previous gradients
    U_total, U_MNL, r = model(X, Q)
    # Decouple gradients for the two parts:
    log_probs_theta = torch.log_softmax(U_MNL + r.detach(), dim=1)
    log_probs_dnn   = torch.log_softmax(U_MNL.detach() + r, dim=1)
    loss_theta = loss_fn(log_probs_theta, y)
    loss_dnn = loss_fn(log_probs_dnn, y)
    loss = loss_theta + loss_dnn
    loss.backward()       # compute gradients
    optimizer.step()      # update parameters

    losses.append(loss.item())
    print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.3f}")


# Compute statistics for interpretable parameters
with torch.no_grad():
    U_int = torch.matmul(X, model.theta).squeeze(2)
    p_int = torch.softmax(U_int, dim=1) 
    p_dim = n_vars
    H = torch.zeros(p_dim, p_dim)
    # For each choice task (session)
    for n in range(num_sessions):
        z_list = []
        # Loop through each alternative in the current session
        for i in range(no_alts):
            z_list.append(X[n, i])
        Z = torch.stack(z_list) 
        #Extract the probability vector for session n and reshape it
        p_n = p_int[n].unsqueeze(1) 
        # Calculate the weighted average of the feature vectors across alternatives, using the choice probabilities as weights → this gives the expected value of attributes
        z_bar = (p_n * Z).sum(dim=0) 
        # Calculate contribution to the Hessian
        for i in range(no_alts):
            diff = Z[i] - z_bar
            H += p_int[n, i] * torch.ger(diff, diff)
    # Regularize the Hessian so it is invertible, this by adding a small constant
    epsilon = 1e-6
    I_obs_reg = H + epsilon * torch.eye(p_dim)
    var_covar_matrix = torch.inverse(I_obs_reg)
    std_errors_theta = torch.sqrt(torch.diag(var_covar_matrix))

# Compute model fit statistics
U_total, _, _ = model(X, Q)
LL_full = -loss_fn(torch.log_softmax(U_total, dim=1), y).item()
LL_null = -num_sessions * np.log(no_alts)
mcfadden_R2 = 1 - (LL_full / LL_null)
# Count total number of parameters: interpretable parameters plus number of learning covariates for adjusted r2 calcs
effective_params = n_vars + n_q
adjusted_R2 = 1 - ((LL_full - effective_params) / LL_null)

# Evaluate performance metrics
with torch.no_grad():
    U_total, _, _ = model(X, Q)
    y_pred = U_total.argmax(dim=1)
y_true = y.numpy()
y_pred_np = y_pred.numpy()

# Calculate acc, precision, recall and F1
accuracy = accuracy_score(y_true, y_pred_np)
precision = precision_score(y_true, y_pred_np, average='macro')
recall = recall_score(y_true, y_pred_np, average='macro')
f1 = f1_score(y_true, y_pred_np, average='macro')

print("\n--- Performance Metrics ---")
print(f"Accuracy: {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1 Score: {f1:.3f}")


# Output results for mnl parameters (the interpretable ones)
theta_hat = model.theta.detach().numpy().flatten() 
z_theta = theta_hat / std_errors_theta.numpy()
p_theta = 2 * (1 - norm.cdf(np.abs(z_theta)))

# Recreate the labels
theta_labels = ["Con"] + [f"{attr}_{level}" for attr in attributes for level in [1, 2]]
theta_df = pd.DataFrame({
    'Coefficient': theta_hat,
    'Std Error': std_errors_theta.numpy(),
    'z-score': z_theta,
    'p-value': p_theta
}, index=theta_labels).round(3)

# Add the model fit statistics
summary_df = pd.DataFrame({
    'Coefficient': [round(LL_full, 3), round(LL_null, 3), round(mcfadden_R2, 3), round(adjusted_R2, 3)],
    'Std Error': [None, None, None, None],
    'z-score': [None, None, None, None],
    'p-value': [None, None, None, None]
}, index=["Log-Likelihood (Full)", "Log-Likelihood (Null)", "McFadden's R2", "McFadden's adjusted R2"])

# Add model performance metrics
metrics_df = pd.DataFrame({
    'Coefficient': [round(accuracy,3), round(precision,3), round(recall,3), round(f1,3)],
    'Std Error': [None, None, None, None],
    'z-score': [None, None, None, None],
    'p-value': [None, None, None, None]
}, index=["Accuracy", "Precision", "Recall", "F1 Score"])

# Combine all results
results_final = pd.concat([theta_df, summary_df, metrics_df])
results_final.to_csv('Lmnl_results.csv', index=True, float_format='%.3f')
print("Results saved to 'Lmnl_results.csv'")

# Plot loss curve
plt.figure(figsize=(8, 6))
plt.plot(range(1, epochs + 1), losses, marker='o', linestyle='-')
plt.title("Training Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()