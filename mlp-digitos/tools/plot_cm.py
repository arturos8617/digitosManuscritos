# tools/plot_cm.py (new file)
import argparse, numpy as np, os
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument('--cm', default='docs/figures/confusion_matrix.csv')
ap.add_argument('--out', default='docs/figures/confusion_matrix.png')
args = ap.parse_args()

cm = np.loadtxt(args.cm, delimiter=',', dtype=int)
plt.figure()
plt.imshow(cm, interpolation='nearest')
plt.title('Confusion Matrix')
plt.colorbar()
ticks = range(10)
plt.xticks(ticks, ticks)
plt.yticks(ticks, ticks)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.savefig(args.out, dpi=150)
print('Saved', args.out)
