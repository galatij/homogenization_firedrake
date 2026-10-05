import numpy as np
import matplotlib.pyplot as plt
import os


# ============================================================
# Settings
# ============================================================

primal_file = "output/primal/Ceff.npy"
mixed_file = "output/mixed/Ceff.npy"

dim = 2


# ============================================================
# Load tensors
# ============================================================

C_primal = np.load(primal_file)
C_mixed = np.load(mixed_file)

print("C_primal shape:", C_primal.shape)
print("C_mixed shape :", C_mixed.shape)


# ============================================================
# Full 2D representation
#
# We retain the unsymmetrized components:
#
# 0 -> 11
# 1 -> 22
# 2 -> 12
# 3 -> 21
#
# This is preferable for debugging the mixed formulation.
# ============================================================

tensor_pairs = [
    (0, 0),
    (1, 1),
    (0, 1),
    (1, 0)
]

labels = [
    "11",
    "22",
    "12",
    "21"
]


def tensor_to_matrix(C):

    n = len(tensor_pairs)

    C_matrix = np.zeros((n, n))

    for I, (i, j) in enumerate(tensor_pairs):

        for J, (k, l) in enumerate(tensor_pairs):

            C_matrix[I, J] = C[i, j, k, l]

    return C_matrix


C_primal_M = tensor_to_matrix(C_primal)
C_mixed_M = tensor_to_matrix(C_mixed)

C_difference_M = C_mixed_M - C_primal_M


# ============================================================
# Print matrices
# ============================================================

np.set_printoptions(
    precision=8,
    suppress=True
)

print("\n========================================")
print("Primal Ceff")
print("========================================")
print(C_primal_M)

print("\n========================================")
print("Mixed Ceff")
print("========================================")
print(C_mixed_M)

print("\n========================================")
print("Mixed - Primal")
print("========================================")
print(C_difference_M)


# ============================================================
# Norms
# ============================================================

norm_primal = np.linalg.norm(C_primal_M)
norm_difference = np.linalg.norm(C_difference_M)

print("\n========================================")
print("Comparison")
print("========================================")

print("||C_primal|| =", norm_primal)

print(
    "||C_mixed - C_primal|| =",
    norm_difference
)

print(
    "Relative error =",
    norm_difference / norm_primal
)


# ============================================================
# Plot
# ============================================================

def plot_matrix(M, title, filename):

    fig, ax = plt.subplots(
        figsize=(7, 6)
    )

    im = ax.imshow(
        M,
        aspect="equal"
    )

    ax.set_xticks(
        range(len(labels))
    )

    ax.set_yticks(
        range(len(labels))
    )

    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    ax.set_xlabel(
        "Gradient component"
    )

    ax.set_ylabel(
        "Stress component"
    )

    ax.set_title(title)

    fig.colorbar(
        im,
        ax=ax,
        label="Tensor component"
    )

    for i in range(M.shape[0]):

        for j in range(M.shape[1]):

            ax.text(
                j,
                i,
                f"{M[i,j]:.3e}",
                ha="center",
                va="center"
            )

    fig.tight_layout()

    fig.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


# ============================================================
# Output
# ============================================================

output_dir = "output/Ceff_comparison"

os.makedirs(
    output_dir,
    exist_ok=True
)


plot_matrix(
    C_primal_M,
    "Ceff — primal",
    os.path.join(
        output_dir,
        "Ceff_primal.png"
    )
)

plot_matrix(
    C_mixed_M,
    "Ceff — mixed",
    os.path.join(
        output_dir,
        "Ceff_mixed.png"
    )
)

plot_matrix(
    C_difference_M,
    "Ceff difference — mixed − primal",
    os.path.join(
        output_dir,
        "Ceff_difference.png"
    )
)