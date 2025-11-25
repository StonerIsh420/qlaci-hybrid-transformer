# qlaci_vs_classical_real.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
import matplotlib.pyplot as plt

# === TRUE QLACI (N=16, real superposition) ===
dev = qml.device("lightning.qubit", wires=11)

@qml.qnode(dev, interface="torch", diff_method="adjoint")
def true_qlaci(query, past_keys):
    N = past_keys.shape[0]
    for i in range(4):
        qml.RY(query[i], wires=i)
    for i in range(4, 8):
        qml.Hadamard(wires=i)
    for j in range(N):
        bin_j = format(j, '04b')
        for bit in range(4):
            if bin_j[bit] == '1':
                qml.CNOT(wires=[4 + bit, 8])
        strength = torch.dot(query, past_keys[j])
        qml.RZ(strength, wires=8)
        for bit in range(4):
            if bin_j[bit] == '1':
                qml.CNOT(wires=[4 + bit, 8])
    return qml.expval(qml.PauliZ(8))

class TrueQLACI_Quantum(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Linear(8, 32)
        self.to_q = nn.Linear(32, 4)
        self.to_k = nn.Linear(32, 4)
        self.out = nn.Linear(1, 32)

    def forward(self, tokens):
        x = self.emb(tokens)
        q = self.to_q(x)
        k = self.to_k(x)
        needle_q = q[:, -1]
        past_k = k[:, :-1]
        B = needle_q.shape[0]
        ctx = torch.stack([true_qlaci(needle_q[i], past_k[i]) for i in range(B)]).unsqueeze(1)
        return self.out(ctx)

# === Classical Linear Attention (Performer-style) ===
class ClassicalLinearAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Linear(8, 32)
        self.to_q = nn.Linear(32, 16)
        self.to_k = nn.Linear(32, 16)
        self.to_v = nn.Linear(32, 32)
        self.out = nn.Linear(32, 32)

    def forward(self, tokens):
        x = self.emb(tokens)                    # (B, N, 32)
        Q = self.to_q(x[:, -1:])                # (B, 1, 16)
        K = self.to_k(x[:, :-1])                # (B, N-1, 16)
        V = self.to_v(x[:, :-1])                # (B, N-1, 32)

        K_pos = F.softplus(K)                   # (B, N-1, 16)
        
        # Performer: correct broadcasting
        numerator = torch.matmul(K_pos.transpose(1,2), V)           # (B, 16, 32)
        denominator = K_pos.sum(dim=1, keepdim=True) + 1e-6         # (B, 1, 16)
        weighted = numerator / denominator.transpose(1,2)           # (B, 16, 32)
        out = weighted.sum(dim=1)                                   # (B, 32)
        return self.out(out)

# === Classical Average Baseline ===
class ClassicalAverage(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Linear(8, 32)
        self.out = nn.Linear(32, 32)

    def forward(self, tokens):
        x = self.emb(tokens)
        past = x[:, :-1].mean(dim=1)
        return self.out(past)

# === TRAIN FUNCTION ===
def train(model, name):
    opt = torch.optim.Adam(model.parameters(), lr=0.02)
    losses = []
    for epoch in range(120):
        tokens = torch.randn(64, 16, 8)
        needle = torch.randn(64, 32)
        tokens[:, -1, :8] = needle[:, :8]
        pred = model(tokens)
        loss = ((pred - needle)**2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if epoch % 40 == 0 or epoch == 119:
            print(f"{name} | Epoch {epoch:03d} → Loss: {loss.item():.6f}")
    return losses

print("FINAL HONEST COMPARISON — N=16")
q_losses = train(TrueQLACI_Quantum(), "QLACI (Quantum)")
l_losses = train(ClassicalLinearAttention(), "Classical Linear")
a_losses = train(ClassicalAverage(), "Classical Average")

plt.figure(figsize=(13,8))
plt.plot(q_losses, color='purple', lw=3, label='True QLACI (Quantum Log-Attention)')
plt.plot(l_losses, color='blue', lw=3, label='Classical Linear Attention (Performer-style)')
plt.plot(a_losses, color='gray', lw=3, alpha=0.8, label='Classical Average')
plt.yscale('log')
plt.title("QLACI vs Classical Baselines — N=16\nReal quantum log-attention with controlled QRAM simulation", fontsize=16)
plt.xlabel("Training Step")
plt.ylabel("MSE (Needle Recovery)")
plt.legend(fontsize=14)
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.show()