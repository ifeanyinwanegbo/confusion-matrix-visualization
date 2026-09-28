import numpy as np
import matplotlib.pyplot as plt

cm = np.array([
    [73, 11, 6],
    [8, 54, 13],
    [4, 17, 96]
])

fig, ax = plt.subplots(figsize=(7, 6))

im = ax.imshow(cm, cmap="viridis")
plt.colorbar(im, ax=ax)

ax.set_title("Multiclass Classification Results")
ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")

ax.set_xticks([0, 1, 2])
ax.set_yticks([0, 1, 2])

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, cm[i, j],
                ha="center", va="center",
                color="black")

plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.show()
