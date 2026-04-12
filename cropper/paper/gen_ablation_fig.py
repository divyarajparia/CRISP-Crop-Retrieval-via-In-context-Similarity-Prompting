#!/usr/bin/env python3
"""Generate ablation IoU bar chart for the paper."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Data
methods = ['Expanded\nR=10', 'Diverse\nICL', 'Multi-\nTemp', 'VILA-\nOnly']
ious = [0.762, 0.758, 0.757, 0.756]
baseline_iou = 0.672

# Colors
colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2']

fig, ax = plt.subplots(figsize=(5, 3.2))

bars = ax.bar(methods, ious, color=colors, width=0.55, edgecolor='white', linewidth=0.8)

# Baseline dashed line
ax.axhline(y=baseline_iou, color='#333333', linestyle='--', linewidth=1.2, label='Cropper baseline (0.672)')

# Value labels on bars
for bar, iou in zip(bars, ious):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
            f'{iou:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Delta labels
for bar, iou in zip(bars, ious):
    delta = iou - baseline_iou
    ax.text(bar.get_x() + bar.get_width()/2, baseline_iou + (iou - baseline_iou)/2,
            f'+{delta:.3f}', ha='center', va='center', fontsize=7.5,
            color='white', fontweight='bold')

ax.set_ylabel('IoU', fontsize=11)
ax.set_ylim(0.64, 0.79)
ax.legend(loc='lower right', fontsize=8.5, framealpha=0.9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='x', labelsize=9)
ax.tick_params(axis='y', labelsize=9)

plt.tight_layout()
plt.savefig('figures/ablation_iou.pdf', dpi=300, bbox_inches='tight')
print("Saved figures/ablation_iou.pdf")
