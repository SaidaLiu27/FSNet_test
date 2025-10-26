import torch
from torch.optim import Adam
import numpy as np
import pandas as pd
import os

from fsnet.model import MLP
from fsnet.layer import fs_unroll
from fsnet.loss import fsnet_total_loss
from safeset.eclippse import violation_loss, rhotheta_violation_loss

D_IN = 10 
D_OUT = 2  
HIDDEN = 64
LR = 1e-3
BATCH_SIZE = 32
EPOCHS = 50 

K = 20 ## total fs steps
KP = 5 # fs steps with grad
ETA = 0.05  # learning rate 
RHO = 1.0  # eq 6, p
Q_THRESHOLD = 0.1 # threshold (section 5.1)
BETA_PENALTY = 10.0 


import safeset.eclippse
safeset.eclippse.rho = BETA_PENALTY

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# for csv
RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)
LOG_FILE = os.path.join(RESULTS_DIR, 'training_log.csv')
log_data = []

model = MLP(D_IN, HIDDEN, D_OUT).to(DEVICE)
optimizer = Adam(model.parameters(), lr=LR)

## state (for ease)
s_state = torch.randn(BATCH_SIZE, D_IN).to(DEVICE)

# ideal output (for ease)
a_star_target = torch.tensor([0.5, 0.0], device=DEVICE).repeat(BATCH_SIZE, 1)

print("--- Training Start ---")

## train 
for epoch in range(EPOCHS):
    
    optimizer.zero_grad()

    a_raw = model(s_state) 

    
    a_hat = fs_unroll(
        a_raw, 
        s_state,
        phi_fn=violation_loss, 
        K=K, Kp=KP, eta=ETA
    )

    total_loss, f_loss, reg_loss, stab_loss = fsnet_total_loss(
        a_raw,
        a_hat,
        a_star_target,
        rho=RHO,
        threshold=Q_THRESHOLD
    )

    total_loss.backward()
    optimizer.step()

    # add log
    with torch.no_grad():
        phi_a_raw_mean = violation_loss(a_raw).mean().item()

    log_entry = {
        'epoch': epoch + 1,
        'total_loss': total_loss.item(),
        'task_loss_f': f_loss.item(),
        'reg_loss_rho': reg_loss.item(),
        'stab_loss_p': stab_loss.item(),
        'raw_violation_phi': phi_a_raw_mean
    }
    log_data.append(log_entry)

    ## --- below : chatGPT ---
    
    if (epoch + 1) % 10 == 0:
        print(f"\n[Epoch {epoch+1}/{EPOCHS}]")
        print(f"  Total Loss: {total_loss.item():.4f}")
        print(f"    ├ Task f(â):      {f_loss.item():.4f}")
        print(f"    ├ Reg ρ||a-â||^2:  {reg_loss.item():.4f}")
        print(f"    └ Stab. P(a_raw): {stab_loss.item():.4f}  <-- Eq. 13 penalty")
        print(f"  Raw Violation (mean φ): {phi_a_raw_mean:.4f} <-- This should go to 0")

df_log = pd.DataFrame(log_data)
df_log.to_csv(LOG_FILE, index=False)
print(f"\nLog saved to {LOG_FILE}")

print("\n--- Training Finished ---")
model.eval()
with torch.no_grad():
    a_raw_final = model(s_state)
    phi_final = violation_loss(a_raw_final)
    
    print(f"Final a* (target):    {a_star_target[0].cpu().numpy()}")
    print(f"Final a_raw (NN pred): {a_raw_final[0].cpu().numpy()}")
    print(f"Final Violation (φ):  {phi_final.mean().item():.6f}")

    if phi_final.mean() < 1e-3:
        print("\nSUCCESS: NN learned to output safe actions directly!")
    else:
        print("\nNOTE: NN output is still violating, FS-Step is still needed.")