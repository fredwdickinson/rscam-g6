import numpy as np
import pandas as pd
import os
from scipy.spatial import distance

def cosine_sim(u, v):
    """ 
    Compute the cosine similarity between two vectors. 
    One (1) is orthogonal, zero (0) is perfect similarity.
    Wrapper around SciPy's similarity for safety.
    """

    # Cast to float64 to minimize rounding errors.
    u = np.asarray(u, dtype = np.float64)
    v = np.asarray(v, dtype = np.float64)

    try:
        return distance.cosine(u, v)
    except ValueError:
        # If we still hit a domain error, it's a precision ghost
        # Return 1.0 (no distance) or 0.0 (perfect similarity) 
        # return 0.0
        print(rf"Vector u: {u}.")
        print(rf"Vector v: {v}.")
        raise ValueError("Error encountered in cosine similarity.")
    
# ==================================================================================
    
def batch_cosine_sim(baseline, synthetic_matrix):
    """ 
    Computes the cosine similarity between a baseline signature and a batch of 
    synthetic signatures.
        Returns a Bx1 vector if synthetic_matrix is Bx96.
    """

    # Cosine similarity is dot(u, v) / (|u| |v|)
    # We are using cosine distance 1 - ... 
    dp = np.dot(synthetic_matrix, baseline) # Shape (B, 1)
    similarity = dp / (np.linalg.norm(synthetic_matrix, axis = 1) * np.linalg.norm(baseline))

    # Clip in case of machine precision.
    return 1.0 - np.clip(similarity, -1.0, 1.0)

# ==================================================================================

def hdi(acceptances, alpha_range, mass = 0.9):
    """ 
    Finds the first interval with a total mass of over mass%.
    (Default 90%, could also use 95%).
    """

    # Retrieves the INDEXES of the highest acceptances.
    sorted_indexes = np.argsort(acceptances)[::-1]
    cumulative_prob = np.cumsum(acceptances[sorted_indexes])

    # Look to see where we first exceed the mass.
    cutoff_index = np.argmax(cumulative_prob >= mass)
    hdi_indexes = sorted_indexes[:cutoff_index + 1] # Takes those inside the HDI.

    # Find the alphas inside the HDI, use to find min, max, mode.
    alphas_hdi = alpha_range[hdi_indexes]
    alpha0 = np.min(alphas_hdi); alpha1 = np.max(alphas_hdi)
    alpha_mode = alpha_range[np.argmax(acceptances)]
    total_mass = np.sum(acceptances[hdi_indexes])

    return alpha0, alpha1, alpha_mode, total_mass

def retrieve_best_tol(est_N_muts_range, csv_path = "../abc_alpha/optimal_abc_tolerances.csv"):
    """ 
    Given an estimated alpha value and an estimated number of mutations, read
    the best tolerance csv file to supply an optimal tolerance value to use.
    """

    path = os.path.abspath(csv_path)
    optimal_tol_df = pd.read_csv(path)

    # Minimise the distance from est_alpha and est_N_muts
    alphas = optimal_tol_df["alpha"].unique() # 10 alphas.
    N_muts_arr = optimal_tol_df["N_muts"].unique() 
    tol_mat = np.zeros((len(est_N_muts_range), len(alphas)), dtype = float)

    # In the df it's called "N_muts" and the best tol is "best_tol".
    for j, est_N_muts in enumerate(est_N_muts_range):
        # For each est_N_muts, select the N_muts in the df that is closest to it.
        closest_N_mut = N_muts_arr[np.argmin(np.abs(N_muts_arr - est_N_muts))]

        for k, alpha in enumerate(alphas):
            best_tol = optimal_tol_df[(optimal_tol_df["N_muts"] == closest_N_mut) & 
                                (optimal_tol_df["alpha"] == alpha)]["best_tol"]
            tol_mat[j, k] = best_tol.iloc[0]

    return tol_mat

