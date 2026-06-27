"""
Generate BES critical point signal plot:
  Left panel:  Mean classifier score vs sqrt(s_NN)
  Right panel: Mean classifier score vs muB

Run:  python3 plot_BES_critical_point_signal.py
"""

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# ── Load trained model ────────────────────────────────────────────────────
ckpt = torch.load('heavyion_classifier.pth', weights_only=False)
FEATURES = ckpt['features']
scaler = StandardScaler()
scaler.mean_ = ckpt['scaler_mean']
scaler.scale_ = ckpt['scaler_scale']

from train_heavyion import HeavyIonClassifier
model = HeavyIonClassifier(n_features=len(FEATURES))
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

# ── BES energy/muB mapping ───────────────────────────────────────────────
bes = {
    7.7:  {'muB': 421, 'csv': 'jets_signal_7gev_smash.csv',  'label': 1},
    11.5: {'muB': 316, 'csv': 'jets_control_11gev_smash.csv', 'label': 1},
    19.6: {'muB': 215, 'csv': 'jets_baseline_19gev_smash.csv','label': 0},
}

energies, muBs, scores = [], [], []

for energy, info in bes.items():
    try:
        df = pd.read_csv(info['csv'])
        X = scaler.transform(df[FEATURES].values.astype(np.float32))
        with torch.no_grad():
            pred = torch.sigmoid(model(torch.FloatTensor(X))).numpy().flatten()
        mean_score = pred.mean()
        energies.append(energy)
        muBs.append(info['muB'])
        scores.append(mean_score)
        print(f"  {energy:>5} GeV  muB={info['muB']} MeV  mean_score={mean_score:.4f}  n_jets={len(df)}")
    except FileNotFoundError:
        print(f"  {energy} GeV: {info['csv']} not found, skipping")

energies = np.array(energies)
muBs     = np.array(muBs)
scores   = np.array(scores)

# ── Figure: two panels ───────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Panel (a): Score vs sqrt(s_NN)
ax1.plot(energies, scores, 'o-', color='crimson', markersize=12, linewidth=2.5,
         markeredgecolor='black', markeredgewidth=1, zorder=5)

# CP region band
ax1.axvspan(7.0, 9.5, alpha=0.15, color='red', label=r'CP region ($\mu_B \approx 350$ MeV)')
ax1.axhline(0.5, color='gray', linestyle='--', linewidth=1, label='Random (0.5)')

for e, s, m in zip(energies, scores, muBs):
    ax1.annotate(f'$\\mu_B={m}$ MeV', (e, s),
                 textcoords='offset points', xytext=(0, 14),
                 ha='center', fontsize=10, fontweight='bold')

ax1.set_xlabel(r'$\sqrt{s_{NN}}$ (GeV)', fontsize=13)
ax1.set_ylabel('Mean Classifier Score', fontsize=13)
ax1.set_title('Critical Point Signal vs Collision Energy', fontsize=13, fontweight='bold')
ax1.set_xlim(5, 22)
ax1.set_ylim(0.2, 0.85)
ax1.legend(loc='upper right', fontsize=10)
ax1.grid(True, alpha=0.3)

# Panel (b): Score vs muB
ax2.plot(muBs, scores, 's-', color='blue', markersize=12, linewidth=2.5,
         markeredgecolor='black', markeredgewidth=1, zorder=5)

ax2.axvline(350, color='green', linestyle='--', linewidth=2, label=r'$\mu_{Bc}=350$ MeV (CP)')
ax2.axhline(0.5, color='gray', linestyle='--', linewidth=1, label='Random')

for m, s, e in zip(muBs, scores, energies):
    ax2.annotate(f'{e} GeV', (m, s),
                 textcoords='offset points', xytext=(0, 14),
                 ha='center', fontsize=10, fontweight='bold')

ax2.set_xlabel(r'$\mu_B$ (MeV)', fontsize=13)
ax2.set_ylabel('Mean Classifier Score', fontsize=13)
ax2.set_title('Score vs Baryon Chemical Potential', fontsize=13, fontweight='bold')
ax2.set_xlim(180, 460)
ax2.set_ylim(0.2, 0.85)
ax2.legend(loc='upper left', fontsize=10)
ax2.grid(True, alpha=0.3)

plt.suptitle('Ising Critical Point Detection via Jet Classifier\n'
             'vHLLE-SMASH + 3D Ising Embedding, Au+Au BES',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('BES_critical_point_signal.png', dpi=150, bbox_inches='tight')
print("\nSaved: BES_critical_point_signal.png")
