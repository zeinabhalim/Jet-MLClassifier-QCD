"""
Afterburner Impact Analysis – Sampler vs SMASH output comparison.
Generates a multi-panel figure showing how the SMASH hadronic
afterburner modifies jet substructure features.

Run:  python3 plot_afterburner_impact.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ── Load data ──────────────────────────────────────────────────────────────
sampler  = pd.read_csv("jets_signal_7gev.csv")
smash    = pd.read_csv("jets_signal_7gev_smash.csv")

print(f"Sampler jets: {len(sampler)}")
print(f"SMASH jets:   {len(smash)}")

# ── Figure ─────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 12))
gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.35)

# ── Panel (a): Jet Mass distribution ───────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
bins_mass = np.linspace(0, 20, 60)
ax1.hist(sampler['jet_mass'], bins=bins_mass, alpha=0.6, density=True,
         color='steelblue', label=f'Sampler (mean={sampler["jet_mass"].mean():.2f})')
ax1.hist(smash['jet_mass'], bins=bins_mass, alpha=0.6, density=True,
         color='crimson', label=f'SMASH (mean={smash["jet_mass"].mean():.2f})')
ax1.set_xlabel('Jet Mass [GeV]', fontsize=11)
ax1.set_ylabel('Density', fontsize=11)
ax1.set_title('(a) Jet Mass', fontsize=12, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# ── Panel (b): sqrt_d12 splitting scale ────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
bins_d12 = np.linspace(0, 10, 60)
mask_s = sampler['sqrt_d12'] > 0
mask_m = smash['sqrt_d12'] > 0
ax2.hist(sampler.loc[mask_s, 'sqrt_d12'], bins=bins_d12, alpha=0.6, density=True,
         color='steelblue', label=f'Sampler (mean={sampler.loc[mask_s,"sqrt_d12"].mean():.2f})')
ax2.hist(smash.loc[mask_m, 'sqrt_d12'], bins=bins_d12, alpha=0.6, density=True,
         color='crimson', label=f'SMASH (mean={smash.loc[mask_m,"sqrt_d12"].mean():.2f})')
ax2.set_xlabel(r'$\sqrt{d_{12}}$ [GeV]', fontsize=11)
ax2.set_ylabel('Density', fontsize=11)
ax2.set_title(r'(b) Splitting Scale $\sqrt{d_{12}}$', fontsize=12, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# ── Panel (c): pT dispersion ────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
bins_ptd = np.linspace(0.5, 1.0, 60)
ax3.hist(sampler['pt_d'], bins=bins_ptd, alpha=0.6, density=True,
         color='steelblue', label=f'Sampler (mean={sampler["pt_d"].mean():.4f})')
ax3.hist(smash['pt_d'], bins=bins_ptd, alpha=0.6, density=True,
         color='crimson', label=f'SMASH (mean={smash["pt_d"].mean():.4f})')
ax3.set_xlabel(r'$p_T$ dispersion $\sqrt{\sum p_{T,i}^2}/\sum p_{T,i}$', fontsize=11)
ax3.set_ylabel('Density', fontsize=11)
ax3.set_title(r'(c) $p_T$ Dispersion', fontsize=12, fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

# ── Panel (d): Jet pT ──────────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 0])
bins_jpt = np.linspace(5, 40, 60)
ax4.hist(sampler['jet_pt'], bins=bins_jpt, alpha=0.6, density=True,
         color='steelblue', label=f'Sampler (mean={sampler["jet_pt"].mean():.1f})')
ax4.hist(smash['jet_pt'], bins=bins_jpt, alpha=0.6, density=True,
         color='crimson', label=f'SMASH (mean={smash["jet_pt"].mean():.1f})')
ax4.set_xlabel(r'$p_{T,jet}$ [GeV]', fontsize=11)
ax4.set_ylabel('Density', fontsize=11)
ax4.set_title(r'(d) Jet $p_T$', fontsize=12, fontweight='bold')
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3)

# ── Panel (e): Jet eta ─────────────────────────────────────────────────────
ax5 = fig.add_subplot(gs[1, 1])
bins_eta = np.linspace(-1.5, 1.5, 60)
ax5.hist(sampler['jet_eta'], bins=bins_eta, alpha=0.6, density=True,
         color='steelblue', label='Sampler')
ax5.hist(smash['jet_eta'], bins=bins_eta, alpha=0.6, density=True,
         color='crimson', label='SMASH')
ax5.set_xlabel(r'$\eta_{jet}$', fontsize=11)
ax5.set_ylabel('Density', fontsize=11)
ax5.set_title(r'(e) Jet $\eta$', fontsize=12, fontweight='bold')
ax5.legend(fontsize=9)
ax5.grid(True, alpha=0.3)

# ── Panel (f): Constituent multiplicity ─────────────────────────────────────
ax6 = fig.add_subplot(gs[1, 2])
bins_nc = np.arange(0, 30, 1)
ax6.hist(sampler['nconstituents'], bins=bins_nc, alpha=0.6, density=True,
         color='steelblue', label=f'Sampler (mean={sampler["nconstituents"].mean():.1f})')
ax6.hist(smash['nconstituents'], bins=bins_nc, alpha=0.6, density=True,
         color='crimson', label=f'SMASH (mean={smash["nconstituents"].mean():.1f})')
ax6.set_xlabel('n constituents', fontsize=11)
ax6.set_ylabel('Density', fontsize=11)
ax6.set_title('(f) Constituent Count', fontsize=12, fontweight='bold')
ax6.legend(fontsize=9)
ax6.grid(True, alpha=0.3)

# ── Panel (g): Percentage shift bar chart ──────────────────────────────────
ax7 = fig.add_subplot(gs[2, 0:2])
features = ['jet_pt', 'jet_mass', 'nconstituents', 'pt_d', 'sqrt_d12']
labels   = [r'$p_{T,jet}$', r'$m_{jet}$', r'$n_{const}$', r'$p_T$ disp.', r'$\sqrt{d_{12}}$']
shifts = []
for f in features:
    s_val = sampler[f].mean()
    m_val = smash[f].mean()
    pct = ((m_val - s_val) / s_val) * 100 if s_val > 0 else 0
    shifts.append(pct)

colors = ['crimson' if s < 0 else 'steelblue' for s in shifts]
bars = ax7.barh(labels, shifts, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
ax7.set_xlabel('Change after SMASH afterburner [%]', fontsize=11)
ax7.set_title('(g) Afterburner Impact on Jet Features', fontsize=12, fontweight='bold')
ax7.axvline(0, color='black', linewidth=1)
ax7.grid(True, alpha=0.3, axis='x')
for bar, val in zip(bars, shifts):
    ax7.text(bar.get_width() + (0.3 if val >= 0 else -0.3),
             bar.get_y() + bar.get_height()/2,
             f'{val:+.1f}%', va='center', ha='left' if val >= 0 else 'right',
             fontsize=10, fontweight='bold')

# ── Panel (h): Summary text box ─────────────────────────────────────────────
ax8 = fig.add_subplot(gs[2, 2])
ax8.axis('off')
summary = (
    "AFTERBURNER IMPACT ANALYSIS\n"
    "Au+Au at 7.7 GeV, 20-50% centrality\n"
    "Ising CP embedding (λ=0.2)\n\n"
    f"Sampler jets:     {len(sampler):>6}\n"
    f"SMASH jets:       {len(smash):>6}\n"
    f"Charged filter:   π±, K±, p\n\n"
    f"Jet mass shift:     "
    f"{((smash['jet_mass'].mean()-sampler['jet_mass'].mean())/sampler['jet_mass'].mean())*100:+.1f}%\n"
    f"√d12 shift:         "
    f"{((smash['sqrt_d12'].mean()-sampler['sqrt_d12'].mean())/sampler['sqrt_d12'].mean())*100:+.1f}%\n"
    f"pT disp shift:      "
    f"{((smash['pt_d'].mean()-sampler['pt_d'].mean())/sampler['pt_d'].mean())*100:+.1f}%\n\n"
    "Blue = Sampler (Cooper-Frye)\n"
    "Red  = After SMASH rescattering"
)
ax8.text(0.05, 0.95, summary, transform=ax8.transAxes,
         fontsize=10, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow',
                   edgecolor='gray', alpha=0.9))

plt.suptitle('Afterburner Impact on Jet Substructure\n'
             'Au+Au 7.7 GeV with 3D Ising CP Embedding (λ=0.2)',
             fontsize=14, fontweight='bold', y=0.98)
plt.savefig('afterburner_impact_analysis.png', dpi=150, bbox_inches='tight')
print("Saved: afterburner_impact_analysis.png")
plt.close()

# ── Also generate a BES energy comparison plot ──────────────────────────────
fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))

try:
    s7  = pd.read_csv("jets_signal_7gev_smash.csv")
    c11 = pd.read_csv("jets_control_11gev_smash.csv")
    b19 = pd.read_csv("jets_baseline_19gev_smash.csv")
except:
    s7  = pd.read_csv("jets_signal_7gev.csv")
    c11 = pd.read_csv("jets_control_11gev.csv")
    b19 = pd.read_csv("jets_baseline_19gev.csv")

# Panel 1: Jet mass comparison across BES energies
axes2[0].hist(s7['jet_mass'], bins=50, alpha=0.5, density=True, color='red', label='7.7 GeV')
axes2[0].hist(c11['jet_mass'], bins=50, alpha=0.5, density=True, color='orange', label='11.5 GeV')
axes2[0].hist(b19['jet_mass'], bins=50, alpha=0.5, density=True, color='blue', label='19.6 GeV')
axes2[0].set_xlabel('Jet Mass [GeV]')
axes2[0].set_ylabel('Density')
axes2[0].set_title('Jet Mass – BES Energies')
axes2[0].legend()
axes2[0].grid(True, alpha=0.3)

# Panel 2: sqrt_d12 comparison
for df, color, lbl in [(s7, 'red', '7.7 GeV'), (c11, 'orange', '11.5 GeV'), (b19, 'blue', '19.6 GeV')]:
    vals = df.loc[df['sqrt_d12'] > 0, 'sqrt_d12']
    axes2[1].hist(vals, bins=50, alpha=0.5, density=True, color=color, label=lbl)
axes2[1].set_xlabel(r'$\sqrt{d_{12}}$ [GeV]')
axes2[1].set_ylabel('Density')
axes2[1].set_title(r'Splitting Scale – BES Energies')
axes2[1].legend()
axes2[1].grid(True, alpha=0.3)

# Panel 3: Mean features as function of beam energy
energies = [7.7, 11.5, 19.6]
for feat, marker, color in [('jet_mass', 'o', 'red'), ('sqrt_d12', 's', 'blue'), ('pt_d', '^', 'green')]:
    means = [s7[feat].mean(), c11[feat].mean(), b19[feat].mean()]
    # Normalize to 7.7 GeV value
    if means[0] > 0:
        norm = [m / means[0] for m in means]
    else:
        norm = means
    axes2[2].plot(energies, norm, marker=marker, color=color, label=feat, linewidth=2, markersize=8)
axes2[2].set_xlabel(r'$\sqrt{s_{NN}}$ [GeV]')
axes2[2].set_ylabel('Normalized mean')
axes2[2].set_title('Feature Evolution Across BES')
axes2[2].legend()
axes2[2].grid(True, alpha=0.3)

plt.suptitle('BES Energy Dependence of Jet Substructure', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('bes_energy_comparison.png', dpi=150, bbox_inches='tight')
print("Saved: bes_energy_comparison.png")
plt.close()
