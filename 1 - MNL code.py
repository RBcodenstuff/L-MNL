import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import norm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

path = "-"
sheet = "-"
df_total = pd.read_excel(path, sheet)

no_alts = 3

attributes = ["-", "-", "-"]
feature_cols = []
for alt in range(1, no_alts + 1):
    feature_cols.append(f"{alt}Con") 
    for attr in attributes:
        for level in [1, 2]:
            feature_cols.append(f"{alt}{attr}_{level}")

print("MNL feature columns:", feature_cols)

X_raw = df_total[feature_cols].values  
num_sessions = X_raw.shape[0]
n_vars = len(feature_cols) // no_alts   

X_np = X_raw.reshape(num_sessions, no_alts, n_vars).astype(np.float32)
for j in range(1, n_vars):
    col = X_np[:, :, j]
    mean_val = np.mean(col)
    std_val = np.std(col)
    if std_val != 0:
        X_np[:, :, j] = (col - mean_val) / std_val
X = torch.tensor(X_np, dtype=torch.float32)
print("X shape:", X.shape) 

learn_vars = ["-", "-", "-"]
q_cols = []
for alt in range(1, no_alts+1):
    for var in learn_vars:
        q_cols.append(f"{alt}{var}")
print("Learning part feature columns:", q_cols)

Q_raw = df_total[q_cols].values  
n_q = len(learn_vars)  
Q_np = Q_raw.reshape(num_sessions, no_alts, n_q).astype(np.float32)
for j in range(n_q):
    col = Q_np[:, :, j]
    mean_val = np.mean(col)
    std_val = np.std(col)
    if std_val != 0:
        Q_np[:, :, j] = (col - mean_val) / std_val
Q_tensor = torch.tensor(Q_np, dtype=torch.float32)
print("Q shape:", Q_tensor.shape)  

X_combined = torch.cat([X, Q_tensor], dim=2)
new_n_vars = n_vars + n_q  
print("Combined X shape:", X_combined.shape)

y = torch.tensor(df_total["choice_"].values, dtype=torch.long) - 1
print("y shape:", y.shape)  # (N,)

class MNLModelShared(nn.Module):
    def __init__(self, no_alts, n_vars):
        super(MNLModelShared, self).__init__()
        self.beta = nn.Parameter(torch.zeros(n_vars))
    
    def forward(self, X):
        utilities = torch.matmul(X, self.beta)  
        return torch.log_softmax(utilities, dim=1)

model = MNLModelShared(no_alts, new_n_vars)

loss_fn = nn.NLLLoss(reduction='sum')
optimizer = optim.LBFGS(model.parameters(), lr=0.1, max_iter=50)

def closure():
    optimizer.zero_grad()
    y_pred = model(X_combined)
    loss = loss_fn(y_pred, y)
    loss.backward()
    return loss

losses = []
epochs = 20
for epoch in range(epochs):
    loss_val = optimizer.step(closure)
    losses.append(loss_val.item())
    print(f"Epoch {epoch+1}/{epochs} - Loss: {loss_val.item():.3f}")

with torch.no_grad():
    utilities = torch.matmul(X_combined, model.beta)  
    p = torch.softmax(utilities, dim=1)             
    p_unsq = p.unsqueeze(2) 
    Ex = (X_combined * p_unsq).sum(dim=1) 
    
    H = torch.zeros(new_n_vars, new_n_vars)
    for i in range(num_sessions):
        xi = X_combined[i]     
        pi = p[i]               
        Ex_i = Ex[i]           
        for j in range(no_alts):
            diff = xi[j] - Ex_i
            H += pi[j] * torch.ger(diff, diff)
    Hessian = -H
    I_obs = H 
    epsilon = 1e-6
    I_obs_reg = I_obs + epsilon * torch.eye(new_n_vars)
    var_covar_matrix = torch.inverse(I_obs_reg)
    std_errors = torch.sqrt(torch.diag(var_covar_matrix))

LL_full = -loss_fn(model(X_combined), y).item()
LL_null = -num_sessions * np.log(no_alts)
mcfadden_R2 = 1 - (LL_full / LL_null)
adjusted_R2 = 1 - ((LL_full - new_n_vars) / LL_null)

with torch.no_grad():
    y_pred_log = model(X_combined)  
    y_pred = y_pred_log.argmax(dim=1)  

y_true = y.numpy()
y_pred_np = y_pred.numpy()

accuracy = accuracy_score(y_true, y_pred_np)
precision = precision_score(y_true, y_pred_np, average='macro')
recall = recall_score(y_true, y_pred_np, average='macro')
f1 = f1_score(y_true, y_pred_np, average='macro')

print("\n--- Performance Metrics ---")
print(f"Accuracy: {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1 Score: {f1:.3f}")

orig_labels = ["Con"] + [f"{attr}_{level}" for attr in attributes for level in [1, 2]]
q_labels = [f"{var}" for var in learn_vars]
param_labels = orig_labels + q_labels

beta_hat = model.beta.detach().numpy()
std_errors_np = std_errors.detach().numpy()
z_scores = beta_hat / std_errors_np
p_values = 2 * (1 - norm.cdf(np.abs(z_scores)))

results_df = pd.DataFrame({
    'Coefficient': beta_hat,
    'Std Error': std_errors_np,
    'z-score': z_scores,
    'p-value': p_values
}, index=param_labels).round(3)

summary_df = pd.DataFrame({
    'Coefficient': [round(LL_full, 3), round(LL_null, 3), round(mcfadden_R2, 3), round(adjusted_R2, 3)],
    'Std Error': [None, None, None, None],
    'z-score': [None, None, None, None],
    'p-value': [None, None, None, None]
}, index=["Log-Likelihood (Full)", "Log-Likelihood (Null)", "McFadden's R2", "McFadden's adjusted R2"])

metrics_df = pd.DataFrame({
    'Coefficient': [round(accuracy, 3), round(precision, 3), round(recall, 3), round(f1, 3)],
    'Std Error': [None, None, None, None],
    'z-score': [None, None, None, None],
    'p-value': [None, None, None, None]
}, index=["Accuracy", "Precision", "Recall", "F1 Score"])

results_final = pd.concat([results_df, summary_df, metrics_df])
results_final.to_csv('mnl_results2.csv', index=True, float_format='%.3f')
print("Results saved to 'mnl_results2.csv'")

plt.figure(figsize=(8, 6))
plt.plot(range(1, epochs + 1), losses, marker='o', linestyle='-')
plt.title("Training Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()
