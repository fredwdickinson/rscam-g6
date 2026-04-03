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

# ==================================================================================

def optimal_alpha_tol(healthy_profile, sbs88_profile, alpha, N_muts_range, 
                      tol_range = None, baseline_comparisons = 5, plot = False):
    """ 
    Provide the mutational profile for a healhy colon, an alpha, and the sbs88 profile.
    Finds the "optimal tolerance" for abc-alpha, where "optimal" minimises the 
        TPR - FPR : how often does it recover signatures derived from alpha, against those from a healthy colon?
    Averages across (5 default) baseline comparison, each with (1000 default) batches.
    Assumes alpha > 0.
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
        tol_best = 0
        J_best = 0 
        
        for tol in tol_range:
            # Store scores across all baseline comparisons, average later.    
            all_J_tol = 0

            for _ in range(baseline_comparisons):
                baseline_healthy_sig = synthetic_data.generate_signature(healthy_profile, N_muts)
                baseline_alpha_sig = synthetic_data.generate_signature(infected_profile, N_muts)

                
                # Run the alpha model with synthetic signatures and infected signatures.
                # 1. Compares alpha-generated to alpha-generated, should return true.
                tp = abc_alpha_classifier(baseline_alpha_sig, healthy_profile, sbs88_profile, 
                                          N_muts, tol = tol)[0]
                
                # 2. Compare healthy to alpha-generated, should return false.
                fp = abc_alpha_classifier(baseline_healthy_sig, healthy_profile, sbs88_profile, 
                                          N_muts, tol = tol)[0]
                
                # Best is T/F, giving 1.
                # F/F Gives 0, T/T gives 0, F/T gives -1.
                all_J_tol += int(tp - fp)

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
        ax.legend(lines, labs, loc = "center right", fontsize = 13)
        ax.set_title(rf"Optimal tolerance against the number of mutations for $\alpha = ${alpha}", fontsize = 14)

        plt.show()

    return best_tols, best_Js

# ==================================================================================

def abc_alpha_inference(baseline_sig, healthy_profile, sbs88_profile, N_muts, hdi_mass = 0.9,
                        alpha_range = None, a_bins = 50, plot = False, tol = 0.25):
    """ 
    Provide a baseline signature. This gives a posterior distribution for alpha, assuming
    the signature is derived from
        alphaSBS88 + (1-alpha)healthyColon.
    The number of mutations N_muts should match the total present in the given signature, but 
    it is also helpful to range this (e.g. for power analysis).
    """

    if (not alpha_range):
        alpha_range = np.linspace(0, 1, a_bins)

    alpha_acceptances = np.zeros_like(alpha_range)
    for j, alpha in enumerate(alpha_range):
        # NOTE Here find the best tolerance for 
        alpha_acceptances[j] = compare_alpha(baseline_sig, healthy_profile, sbs88_profile,
                                             alpha, N_muts, tol = tol)

    # Normalise the acceptances. 
    if not (np.sum(alpha_acceptances) > 1e-6):
        warnings.warn(rf"No value of $\alpha$ was accepted. Try adjusting the tolerance. Here tol = {tol} with {N_muts} mutations.")
        # Zeroes are a0, a1, mode.
        return alpha_acceptances, alpha_range, 0, 0, 0, hdi_mass
    
    alpha_acceptances = alpha_acceptances / np.sum(alpha_acceptances)
    if (np.abs(1 - np.sum(alpha_acceptances)) > 1e-6):
        raise ValueError(r"Error in normalising the posterior distribution on $\alpha$.")
    
    # Find the interval of alphas corresponding to 90% (default) mass.
    # Uses the hdi() function below.
    alpha0, alpha1, alpha_mode, total_mass = hdi(alpha_acceptances, alpha_range, mass = hdi_mass)
    
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
    return alpha_acceptances, alpha_range, alpha0, alpha1, alpha_mode, hdi_mass

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

def abc_alpha_classifier(baseline_sig, healthy_profile, sbs88_profile, N_muts, 
                         hdi_mass = 0.9, tol = 0.25, plot = False):
    """ 
    Provide a baseline signature. This classifies the signature as having risen from 
        an infected colon alphaSBS88 + (1-alpha)healthyColon.
    The tolerance is very important and should be tuned. Classification logic as discussed notebook.
    Returns: 
        sbs88 detected? :: boolean
        approximate injection level :: float
    """

    acceptances, alpha_range, a0, a1, a_mode, hdi_mass = abc_alpha_inference(baseline_sig, healthy_profile,
                                                            sbs88_profile, N_muts, hdi_mass = hdi_mass, tol = tol, plot = plot)

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