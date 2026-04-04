import numpy as np
from pathlib import Path
from scipy.spatial import distance
import synthetic_data
import warnings

import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["DejaVu Serif"]


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

def compare_models(baseline_sig, assumed_profile, N_muts, tol, f_comparison = cosine_sim, B = 1000):
    """ 
    Provide a baseline signature and test the hypothesis
        "this signature has been derived from a particular set of weights".
        e.g. this signature could arise from 0.4SBS1 + 0.4SBS5 + 0.2SBS18.

    Specify the number of mutations present in the synthetically generated signatures. 
    Specify the tolerance to NOTE "accept" a signature being similar. 
    """

    accepts = np.zeros(B)
    for j in range(B):
        synthetic_sig = synthetic_data.generate_signature(assumed_profile, N_muts)
        difference_score = f_comparison(baseline_sig, synthetic_sig)

        # 
        if (np.abs(difference_score) < tol):
            accepts[j] = 1

    percentage_accept = np.sum(accepts) / B 
    return percentage_accept

# ==================================================================================

def compare_alpha(baseline_sig, healthy_profile, sbs88_profile, alpha, N_muts,
                   tol, B = 1000, distances_only = False):
    """ 
    Provide an injection level alpha, a baseline signature, and test the hypothesis
        "this signature has been derived with alpha = alpha 
        under the mutational profile alphaSBS88 + (1-alpha)healthyColon." 

    Specify the number of mutations present in the synthetic signatures. 
    NOTE These should match the baseline signature, if it's synthetically generated. 
    Specify the tolerance at which we accept an alpha. Low tolerance: harsher acceptance rate.
        Newer version with vectorisation.
    """

    # Generate infected profile for that alpha.
    profile = synthetic_data.composite_profile(healthy_profile, sbs88_profile, alpha)

     # Compare all at once.
    synthetic_sig_mat = synthetic_data.generate_signature(profile, N_muts, B = B)
    difference_scores = batch_cosine_sim(baseline_sig, synthetic_sig_mat)

    if (distances_only):
        # No tol used, returns a 1000x1 vector (B = 1000).
        return difference_scores
    
    percentage_accept = np.mean(difference_scores < tol)
    return percentage_accept

# ==================================================================================


def optimal_alpha_tol(healthy_profile, sbs88_profile, alpha, N_muts_range, 
                      tol_range = None, baseline_comparisons = 100, plot = False):
    """ 
    Provide the mutational profile for a healhy colon, an alpha, and the sbs88 profile.
    Finds the "optimal tolerance" for abc-alpha, where "optimal" minimises the 
        TPR - FPR : how often does it recover signatures derived from alpha, against those from a healthy colon?
    Averages across (5 default) baseline comparison, each with (1000 default) batches.
    """

    # J refers to TPR - FPR.
    best_tols = np.zeros_like(N_muts_range, dtype = np.float64)
    best_Js = np.zeros_like(N_muts_range, dtype = np.float64)
    infected_profile = synthetic_data.composite_profile(healthy_profile, sbs88_profile, alpha)

    if (not tol_range.any()):
        # 0.025, 0.05, 0.075, ..., 0.4
        tol_range = np.arange(0.025, 0.4 + 0.025, 0.025)

    # For each number of mutations, try each tolerance value.
    # Average it across the number of baseline comparisons.
    for j, N_muts in enumerate(N_muts_range):
        # Generate all num_comparisons lot of the healthy/alpha signatures
        baseline_healthy_sigs = np.array([synthetic_data.generate_signature(healthy_profile, N_muts) for _ in range(baseline_comparisons)])
        baseline_infected_sigs = np.array([synthetic_data.generate_signature(infected_profile, N_muts) for _ in range(baseline_comparisons)])
        
        # compare_alpha() will give the cosine difference from two mutational signatures.
        # Want the distances for each value of alpha, for each baseline signature, across all B=1000 batches.
        a_range = np.linspace(0, 1, 100) # Uses a_bins = 100.
        distances_tp = np.zeros((len(a_range), baseline_comparisons, 1000))
        distances_fp = np.zeros((len(a_range), baseline_comparisons, 1000))

        # Create distance matrix.
        for i, alpha_abc in enumerate(a_range):
            infected_profile = synthetic_data.composite_profile(healthy_profile, sbs88_profile, alpha_abc)
            synthetic_comparisons_mat = synthetic_data.generate_signature(infected_profile, N_muts, B = 1000)

            for k in range(baseline_comparisons):
                # Compare infected signature across a range of alphas.
                distances_tp[i, k, :] = batch_cosine_sim(baseline_infected_sigs[k], synthetic_comparisons_mat)
                distances_fp[i, k, :] = batch_cosine_sim(baseline_healthy_sigs[k], synthetic_comparisons_mat)
                
        # Shape of both: 50 x baseline_comparisons x B
        # Normally 50 x 100 x 1000. Each row corresponds to an alpha.
        # Now compare across a range of tolerances: not recompute every time.
        best_tol = 0; best_J = 0;
        for tol in tol_range:
            # Calculate posteriors for all 100 baselines across all 50 alphas.
            # Shape 50x100, mean is over [alpha, baseline, :]
            tp_posteriors = np.mean(distances_tp < tol, axis = 2)
            fp_posteriors = np.mean(distances_fp < tol, axis = 2) 
            
            # Normalise, 1e-12 for avoiding div-by-zero (won't warn here).
            # NOTE Later look to reintroducing the warnings.
            tp_posteriors_norm = tp_posteriors / (tp_posteriors.sum(axis = 0) + 1e-12)
            fp_posteriors_norm = fp_posteriors / (fp_posteriors.sum(axis = 0) + 1e-12)

            tp_count = 0; fp_count = 0;
            # Classify every the alphas from every baseline signature comparison using hdi().
            for k in range(baseline_comparisons):     
                # Check enough have been accepted.
                # NOTE important! Normalisation blows up the ratios.
                total_tp_accepted = np.sum(tp_posteriors[:, k])*1000
                total_fp_accepted = np.sum(fp_posteriors[:, k])*1000

                if (total_tp_accepted > 10):
                    a0_tp, a1_tp, mode_tp, _ = hdi(tp_posteriors_norm[:, k], a_range)
                    tp_count += abc_alpha_classifier(a0_tp, a1_tp, mode_tp)[0]
                
                if (total_fp_accepted > 10):
                    a0_fp, a1_fp, mode_fp, _ = hdi(fp_posteriors_norm[:, k], a_range)
                    fp_count += abc_alpha_classifier(a0_fp, a1_fp, mode_fp)[0]

            # Best J = TPR - FPR, averaged across the number of baseline_comparisons.
            tpr = tp_count / baseline_comparisons
            fpr = fp_count / baseline_comparisons
            # print(f"J score is {round(tpr - fpr, 4)} for tol {round(tol, 3)} and muts {N_muts}.")
            J_tol = tpr - fpr

            if (J_tol > best_J):
                best_tol = tol 
                best_J = J_tol 
            
        # Outside tolerance loop.
        best_tols[j] = best_tol 
        best_Js[j] = best_J

    if (plot):
        fig, ax = plt.subplots(figsize = (8, 4), tight_layout = True)
        line1 = ax.plot(N_muts_range, best_tols, marker = "x", color = "red", label = "Tolerance")
        ax.set_xlabel("Number of Mutations", fontsize = 12)
        ax.set_ylabel("Tolerance", fontsize = 12)

        ax2 = ax.twinx()
        line2 = ax2.plot(N_muts_range, best_Js, marker = "^", color = "blue", label = "TPR - FPR")
        ax2.set_ylabel(r"$J$-Statistic (TPR - FPR)", fontsize = 12)
        ax2.set_ylim(-0.05, 1.05)

        # For combining the legend.
        lines = line1 + line2
        labs = [l.get_label() for l in lines]
        ax.legend(lines, labs, loc = "center right", fontsize = 13)
        ax.set_title(rf"Optimal tolerance against the number of mutations for $\alpha = ${alpha}", fontsize = 14)

        plt.savefig(f"optimal-alpha-{alpha}.png", format = "png", dpi = 200)
        plt.show()

    return best_tols, best_Js

# ==================================================================================

def abc_alpha_inference(baseline_sig, healthy_profile, sbs88_profile, N_muts, hdi_mass = 0.9,
                        alpha_range = None, a_bins = 100, plot = False, tol = 0.05):
    """ 
    Provide a baseline signature. This gives a posterior distribution for alpha, assuming
    the signature is derived from
        alphaSBS88 + (1-alpha)healthyColon.
    The number of mutations N_muts should match the total present in the given signature, but 
    it is also helpful to range this (e.g. for power analysis).
        Returns the distribution, the alpha range, a0, a1, a_mode, and the hdi_mass.
    """

    if (not alpha_range):
        alpha_range = np.linspace(0, 1, a_bins)

    alpha_acceptances = np.array([compare_alpha(baseline_sig, healthy_profile, sbs88_profile,
                                             alpha, N_muts, tol = tol) for alpha in alpha_range])

    # Normalise the acceptances. 
    total_accepted_samples = np.sum(alpha_acceptances)*1000
    if total_accepted_samples < 10:
        # Not enough data to form a reliable posterior. 
        return alpha_acceptances, alpha_range, 0.0, 0.0, 0.0, hdi_mass, False, 0.0
    
    alpha_acceptances = alpha_acceptances / np.sum(alpha_acceptances)
    if (np.abs(1 - np.sum(alpha_acceptances)) > 1e-6):
        raise ValueError(r"Error in normalising the posterior distribution on $\alpha$.")
    
    # Find the interval of alphas corresponding to 90% (default) mass.
    # Uses the hdi() function below.
    alpha0, alpha1, alpha_mode, total_mass = hdi(alpha_acceptances, alpha_range, mass = hdi_mass)
    sbs88_detected, estimated_alpha = abc_alpha_classifier(alpha0, alpha1, alpha_mode)
    
    if (plot):
        fig, ax = plt.subplots(figsize = (8, 4), tight_layout = True)
        ax.plot(alpha_range, alpha_acceptances, marker = 'x', markersize = 4, color = 'blue',
                 alpha = 0.75, label = "Posterior")
        
        ax.fill_between(alpha_range, alpha_acceptances, 
                        where = (alpha_range >= alpha0) & (alpha_range <= alpha1), 
                        color = 'lightblue', alpha=0.85, label = f"{hdi_mass*100:.0f}% HDI")
        
        ax.set_xlabel(r"Injection Level $\alpha$", fontsize = 12)
        ax.set_ylabel("Normalised Acceptance Probability", fontsize = 12)
        ax.set_title(r"ABC Posterior Distribution for the Injection Level $\alpha$", fontsize = 14)
        ax.legend(loc = "center right", fontsize = 13)

        plt.show()

    # NOTE Currently don't use the total mass, just the density mass specified (90%).
    return alpha_acceptances, alpha_range, alpha0, alpha1, alpha_mode, hdi_mass, sbs88_detected, estimated_alpha

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


# ====================================================================================

def abc_alpha_classifier(a0, a1, a_mode):
    """ 
    Provide a baseline signature. This classifies the signature as having risen from 
        an infected colon alphaSBS88 + (1-alpha)healthyColon.
    The tolerance is very important and should be tuned. Classification logic as discussed notebook.
    Returns: 
        sbs88 detected? :: boolean
        approximate injection level :: float
    """
    # First check to see if alpha0 = alpha1, then deny.
    if (np.abs(a0 - a1) < 0.01):
        return False, 0.0
    
    # Case 1: alpha0 > 0.025, then imply SBS88 presence.
    if (a0 > 0.025):
        return True, a_mode
    # Case 2: alpha0 = 0, then reject SBS88 presence.
    elif (np.abs(a0) < 1e-8):
        return False, 0.0
    # Case 3: split into two cases.
    else:                                                         
        # a1 > 0.1 AND mode is > 0.05
        if (a1 > 0.1) and (a_mode > 0.05):
            return True, a_mode
        else:
            return False, 0.0
        
# ====================================================================================

def detect_sbs88(baseline_sig, healthy_profile, sbs88_profile):
    """
    Give a baseline signature.
    Detect the presence of SBS88 assuming it has [3, 5, 10, 100, 200, 500] mutations.
    """

    N_muts_range = np.array([3, 5, 10, 100, 200, 500])
    
    # NOTE Change this from being hard coded later.
    best_tol_list = [0.15, 0.15, 0.175, 0.1, 0.075, 0.075]

    if (len(best_tol_list) != len(N_muts_range)):
        raise ValueError("Mismatch in mutation range/tolerance range shapes.")

    detected = np.zeros_like(N_muts_range, dtype = bool)
    estimated_alphas = np.zeros_like(N_muts_range, dtype = float)

    for j, best_tol in enumerate(best_tol_list):
        N_muts = N_muts_range[j]
        all_results = abc_alpha_inference(baseline_sig, healthy_profile, sbs88_profile,
                                                        N_muts, tol = best_tol)
        
        sbs88_detected, alpha_est = all_results[-2:]
        detected[j] = sbs88_detected
        estimated_alphas[j] = alpha_est 
    
    return detected, estimated_alphas, N_muts_range


# ====================================================================================
# ====================================================================================
# ====================================================================================
# ====================================================================================
# ====================================================================================


def old_abc_alpha_inference(baseline_sig, healthy_profile, sbs88_profile, N_muts, hdi_mass = 0.9,
                        alpha_range = None, a_bins = 50, plot = False, tol = 0.25):
    """ 
    Provide a baseline signature and provide a posterior distribution for alpha, assuming
    the signature is derived from
        alphaSBS88 + (1-alpha)healthyColon.
    NOTE Old - since have added highest density intervals and classifiers.
    Only still here so the notebook makes sense.
    """

    if (not alpha_range):
        alpha_range = np.linspace(0, 1, a_bins)

    alpha_acceptances = np.zeros_like(alpha_range)
    for j, alpha in enumerate(alpha_range):
        # NOTE Here find the best tolerance for 
        alpha_acceptances[j] = compare_alpha(baseline_sig, healthy_profile, sbs88_profile,
                                             alpha, N_muts, tol = tol)
    
    alpha_acceptances = alpha_acceptances / np.sum(alpha_acceptances)
    if (np.abs(1 - np.sum(alpha_acceptances)) > 1e-6):
        raise ValueError(r"Error in normalising the posterior distribution on $\alpha$.")
    
    # Find the interval of alphas corresponding to 90% (default) mass.
    # Uses the hdi() function below.
    
    if (plot):
        fig, ax = plt.subplots(figsize = (8, 4), tight_layout = True)
        ax.plot(alpha_range, alpha_acceptances, marker = 'x', markersize = 4, color = 'blue',
                 alpha = 0.75, label = "Posterior")

        
        ax.set_xlabel(r"Injection Level $\alpha$", fontsize = 12)
        ax.set_ylabel("Normalised Acceptance Probability", fontsize = 12)
        ax.set_title(r"ABC Posterior Distribution for the Injection Level $\alpha$", fontsize = 14)
        ax.legend(loc = "center right", fontsize = 13)

        plt.show()

    # NOTE Currently don't use the total mass, just the density mass specified (90%).
    return alpha_acceptances, alpha_range

# 

def old_optimal_alpha_tol(healthy_profile, sbs88_profile, alpha, N_muts_range, 
                      tol_range = None, baseline_comparisons = 5, B = 1000, plot = False):
    """ 
    Provide the mutational profile for a healhy colon, an alpha, and the sbs88 profile.
    Finds the "optimal tolerance" for abc-alpha, where "optimal" minimises the 
        TPR - FPR : how often does it recover signatures derived from alpha, against those from a healthy colon?
    Averages across (5 default) baseline comparison, each with (1000 default) batches.
    Assumes alpha > 0.
    NOTE This is also old, just here so the notebook makes sense.
    """

    # J refers to TPR - FPR.
    best_tols = np.zeros_like(N_muts_range, dtype = np.float64)
    best_Js = np.zeros_like(N_muts_range, dtype = np.float64)
    infected_profile = synthetic_data.composite_profile(healthy_profile, sbs88_profile, alpha)

    if (not tol_range):
        # 0.025, 0.05, 0.075, ..., 0.4
        tol_range = np.arange(0.025, 0.4 + 0.025, 0.025)

    # For each number of mutations, try each tolerance value.
    # Average it across the number of baseline comparisons.
    for j, N_muts in enumerate(N_muts_range):
        tol_best = 0
        J_best = 0 
        
        for tol in tol_range:
            # Store scores across all baseline comparisons, average later.    
            all_J_tol = 0

            for _ in range(baseline_comparisons):
                baseline_healthy_sig = synthetic_data.generate_signature(healthy_profile, N_muts)
                baseline_alpha_sig = synthetic_data.generate_signature(infected_profile, N_muts)

                
                # Run the alpha model with synthetic signatures and infected signatures.
                tpr_score = compare_alpha(baseline_alpha_sig, healthy_profile, sbs88_profile, 
                                          alpha, N_muts, tol = tol) # Compares alpha-generated to alpha-generated.
                fpr_score = compare_alpha(baseline_healthy_sig, healthy_profile, sbs88_profile, 
                                          alpha, N_muts, tol = tol) # Compares healthy to alpha-generated.
                
                # Ideally TPR is high and FPR is low.
                J = tpr_score - fpr_score 
                all_J_tol += J 

            # After looping over all baseline comparisons, average.
            J_tol = all_J_tol / baseline_comparisons 
            if (J_tol > J_best):
                J_best = J_tol 
                tol_best = tol
        
        # Store the best tolerance value for that number of mutations.
        best_tols[j] = tol_best 
        best_Js[j] = J_best 

    if (plot):
        fig, ax = plt.subplots(figsize = (8, 4), tight_layout = True)
        line1 = ax.plot(N_muts_range, best_tols, marker = "x", color = "red", label = "Tolerance")
        ax.set_xlabel("Number of Mutations", fontsize = 12)
        ax.set_ylabel("Tolerance", fontsize = 12)

        ax2 = ax.twinx()
        line2 = ax2.plot(N_muts_range, best_Js, marker = "^", color = "blue", label = "TPR - FPR")
        ax2.set_ylabel(r"$J$-Statistic (TPR - FPR)", fontsize = 12)

        # For combining the legend.
        lines = line1 + line2
        labs = [l.get_label() for l in lines]
        ax.legend(lines, labs, loc = "upper center", fontsize = 13)
        ax.set_title(rf"Optimal tolerance against the number of mutations for $\alpha = ${alpha}", fontsize = 14)

        plt.show()

    return best_tols, best_Js

def old_compare_alpha(baseline_sig, healthy_profile, sbs88_profile, alpha, N_muts,
                   tol = 0.25, f_comparison = cosine_sim, B = 1000):
    """ 
    Provide an injection level alpha, a baseline signature, and test the hypothesis
        "this signature has been derived with alpha = alpha 
        under the mutational profile alphaSBS88 + (1-alpha)healthyColon." 

    Specify the number of mutations present in the synthetic signatures. 
    NOTE These should match the baseline signature, if it's synthetically generated. 
    Specify the tolerance at which we accept an alpha. Low tolerance: harsher acceptance rate.
    """

    accepts = np.zeros(B)
    profile = synthetic_data.composite_profile(healthy_profile, sbs88_profile, alpha)

    for j in range(B):
        synthetic_sig = synthetic_data.generate_signature(profile, N_muts)
        difference_score = f_comparison(baseline_sig, synthetic_sig)

        if (np.abs(difference_score) < tol):
            accepts[j] = 1

    percentage_accept = np.sum(accepts) / B 
    return percentage_accept