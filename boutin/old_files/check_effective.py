import numpy as np
from firedrake import *
import matplotlib.pyplot as plt

def check_effective_tensors(
        dim,
        Ceff0,
        tCeff0,
        CST=None,
        cst=None,
        mesh=None,
        verbose=True,
        plot=False):
    """
    Post-process homogenized fourth-order tensors.

    Parameters
    ----------
    Ceff0 : ndarray
        Effective elasticity tensor C^0_eff, shape (dim,dim,dim,dim).

    tCeff0 : ndarray
        Effective tilde tensor, shape (dim,dim,dim,dim).

    CST : optional
        Reference isotropic tensor. Can be numpy or UFL/Firedrake tensor.

    cst : optional
        Local elasticity tensor (UFL/Firedrake).
        Used only if its average is requested.

    mesh : optional
        Needed if cst is a Firedrake expression.

    Returns
    -------
    dict
        Dictionary containing errors, Voigt matrices, effective Lamé
        parameters, anisotropy measures.
    """

    results = {}

    if dim not in (2, 3):
        raise ValueError(f"Unsupported dimension {dim}")
    
    # Convert effective tensors
    # Ceff0 = _dict_to_tensor4(Ceff0, dim)
    # tCeff0 = _dict_to_tensor4(tCeff0, dim)

    # Voigt conversion

    Ceff_voigt = tensor_to_voigt(Ceff0, dim)

    results["Ceff_voigt"] = Ceff_voigt

    # Compare with CST
    if CST is not None:
        try:
            CST_np = np.asarray(CST)
            CSTsym_np = 0.5 * (CST_np + np.swapaxes(CST_np, 2, 3))

        except Exception:
            raise TypeError(
                "CST is not a numpy tensor. "
                "Evaluate it before calling this function."
            )

        if verbose:
            print("\nComparison with CST")
            print("-------------------")
            print("||Ceff-CST||/||CST||            = ", np.linalg.norm(Ceff0 - CST_np)/np.linalg.norm(CST_np))
            print("||Ceff-CSTsym||/||CSTsym||      = ", np.linalg.norm(Ceff0 - CSTsym_np)/np.linalg.norm(CSTsym_np))

    # Visualization
    if plot:

        # ==========================================================
        # Figure 1
        # ==========================================================
        Ceff_voigt = tensor_to_voigt(Ceff0, dim)
        CST_voigt = tensor_to_voigt(CST_np, dim)
        CSTsym_voigt = tensor_to_voigt(CSTsym_np, dim)
        

        matrices = [
            (Ceff_voigt, "Ceff"),
            (CST_voigt, "CST"),
            (CSTsym_voigt, "CSTsym"),
        ]

        fig, axes = plt.subplots(1, 3, figsize=(10, 8))

        vmin = min(np.min(M) for M, _ in matrices)
        vmax = max(np.max(M) for M, _ in matrices)

        for ax, (M, title) in zip(axes.flat, matrices):
            im = ax.imshow(
                M,
                interpolation="nearest",
                vmin=vmin,
                vmax=vmax
            )
            ax.set_title(title)
            ax.set_xlabel("Voigt index")
            ax.set_ylabel("Voigt index")
            fig.colorbar(im, ax=ax)

        fig.suptitle("Voigt representations", fontsize=14)
        fig.tight_layout()
        plt.show()


        # ==========================================================
        # Figure 2
        # ==========================================================

        differences = [
            (
                Ceff_voigt - CST_voigt,
                r"$C_{\mathrm{eff}} - C^{\mathrm{ST}}$"
            ),
            (
                Ceff_voigt - CSTsym_voigt,
                r"$C_{\mathrm{eff}} - C^{\mathrm{ST}}_{\mathrm{sym}}$"
            )
        ]

        fig, axes = plt.subplots(1, 2, figsize=(10, 8))

        # Same symmetric scale around zero for all error plots
        max_abs = max(np.max(np.abs(M)) for M, _ in differences)

        for ax, (M, title) in zip(axes.flat, differences):
            im = ax.imshow(
                M,
                interpolation="nearest",
                vmin=-max_abs,
                vmax=max_abs
            )

            ax.set_title(title)
            ax.set_xlabel("Voigt index")
            ax.set_ylabel("Voigt index")
            fig.colorbar(im, ax=ax)

        fig.suptitle(
            "Differences between effective and reference tensors",
            fontsize=14
        )

        fig.tight_layout()
        plt.show()


    # # Isotropic projection
    # lambda_eff = (
    #     Ceff0[0,0,1,1]
    #     +
    #     Ceff0[0,0,2,2]
    #     +
    #     Ceff0[1,1,2,2]
    # )/3


    # mu_eff = (
    #     Ceff0[0,1,0,1]
    #     +
    #     Ceff0[0,2,0,2]
    #     +
    #     Ceff0[1,2,1,2]
    # )/3


    # results["lambda_eff"] = lambda_eff
    # results["mu_eff"] = mu_eff


    # delta = np.eye(3)

    # Ciso = np.zeros((3,3,3,3))

    # for i in range(3):
    #     for j in range(3):
    #         for k in range(3):
    #             for l in range(3):

    #                 Ciso[i,j,k,l] = (
    #                     mu_eff*delta[i,k]*delta[j,l]
    #                     +
    #                     lambda_eff*delta[i,j]*delta[k,l]
    #                 )


    # anisotropy = (
    #     np.linalg.norm(Ceff0-Ciso)
    #     /
    #     np.linalg.norm(Ceff0)
    # )

    # results["anisotropy"] = anisotropy


    # if verbose:

    #     print("\nEffective isotropic parameters")
    #     print("--------------------------------")
    #     print("lambda_eff =", lambda_eff)
    #     print("mu_eff     =", mu_eff)

    #     # print("\nAnisotropy")
    #     # print("--------------------------------")
    #     # print(
    #     #     "||Ceff-Ciso||/||Ceff|| = ",
    #     #     anisotropy
    #     # )


    return results

def _dict_to_tensor4(Cdict, dim):
    C = np.zeros((dim,dim,dim,dim))

    for (l, m), C_lm in Cdict.items():
        C[:, :, l, m] = np.asarray(C_lm)

    return C

def tensor_to_voigt(C, dim):
    if dim == 2:
        pairs = [(0,0), (1,1), (0,1)]
    else:
        pairs = [(0,0), (1,1), (2,2), (1,2), (0,2), (0,1)]

    n = len(pairs)
    V = np.zeros((n, n))

    for I, (i, j) in enumerate(pairs):
        for J, (k, l) in enumerate(pairs):
            V[I, J] = C[i, j, k, l]

    return V