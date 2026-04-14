import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial import distance
import synthetic_data
import helpers
import warnings

import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["DejaVu Serif"]


# ==================================================================================

def compare_alpha(baseline_sig, healthy_profile, sbs88_profile, alpha, N_muts,
                   tol, B = 1000):
    """ 
    Provide an injection level alpha, a baseline signature, and test the hypothesis
        "this signature has been derived with alpha = alpha 
        under the mutational profile alphaSBS88 + (1-alpha)healthyColon." 

    Specify the number of mutations present in the synthetic signatures and the tolerance to test at.
    NOTE This should roughly roughly match the baseline signature, if it's synthetically generated, but
    for power analysis it can vary. Low tolerance: harsher acceptance rate.    

    Returns: percentange of synthetically generated profiles (for the given alpha) that match the baseline.
    """

    # Generate infected profile for that alpha.
    profile = synthetic_data.composite_profile(healthy_profile, sbs88_profile, alpha)

     # Compare all synthetic signatures at once using batch_cosine_sim.
    synthetic_sig_mat = synthetic_data.generate_signature(profile, N_muts, B = B)
    difference_scores = helpers.batch_cosine_sim(baseline_sig, synthetic_sig_mat)
    
    percentage_accept = np.mean(difference_scores < tol)
    return percentage_accept

# ==================================================================================


def abc_alpha_inference(baseline_sig, healthy_profile, sbs88_profile, N_muts, hdi_mass = 0.9,
                         a_bins = 100, tol = 0.05, B = 1000, plot = False, ax = None, show = True, title = None):
    """ 
    Provide a baseline signature. This gives a posterior distribution for alpha, assuming
    the signature is derived from
        alphaSBS88 + (1-alpha)healthyColon.
    The number of mutations N_muts should match the total present in the given signature, but 
    it is also helpful to range this (e.g. for power analysis).
        Returns the distribution, the alpha range, a0, a1, a_mode, and the hdi_mass.
    """


    # Percentage of synthetically generated signatures that were accepted for each alpha in the alpha bins.
    alpha_range = np.linspace(0, 1, a_bins)
    alpha_acceptances = np.array([compare_alpha(baseline_sig, healthy_profile, sbs88_profile,
                                             alpha, N_muts, tol = tol, B = B) for alpha in alpha_range])

    # Normalise the acceptances. 
    total_accepted_samples = np.sum(alpha_acceptances)*B
    if total_accepted_samples < int(B/200): # Default to 5 if B = 1000.
        # Get uniform dist and thus alpha_mode = 0, interval [0, 1]
        hdi_mass = 1
        alpha0, alpha1, alpha_mode, total_mass = helpers.hdi(np.zeros_like(alpha_range), alpha_range, mass = hdi_mass)
    else:
        # Normalised acceptance rate.
        alpha_acceptances = alpha_acceptances / np.sum(alpha_acceptances)
        
        # Find the interval of alphas corresponding to 90% (default) mass. Uses hdi().
        alpha0, alpha1, alpha_mode, total_mass = helpers.hdi(alpha_acceptances, alpha_range, mass = hdi_mass)
    
    if (plot):
        if (not ax):
            fig, ax = plt.subplots(figsize = (8, 4), tight_layout = True)
        
        ax.plot(alpha_range, alpha_acceptances, marker = 'x', markersize = 4, color = 'blue',
                 alpha = 0.75, label = "Posterior")
        
        ax.plot(alpha_mode, alpha_acceptances[np.where(alpha_range == alpha_mode)], marker = 'o', 
                markersize = 5, color = 'black', alpha = 0.85, label = rf"Est. $\alpha = ${round(alpha_mode, 2)}")
        
        if (hdi_mass != 1):
            ax.fill_between(alpha_range, alpha_acceptances, 
                            where = (alpha_range >= alpha0) & (alpha_range <= alpha1), 
                            color = 'lightblue', alpha=0.85, label = f"{hdi_mass*100:.0f}% HDI")
        
        ax.set_xlabel(r"Injection Level $\alpha$", fontsize = 14)
        ax.set_ylabel("Density", fontsize = 14)
        
        if (not title):
            title = rf"ABC Posterior Distribution for the Injection Level $\alpha$, tol = {round(tol, 4)}"
        ax.set_title(title, fontsize = 16)

        # Adjustment for the uniform
        legend_loc = "upper center" if hdi_mass == 1 else "center right"
        ax.legend(loc = legend_loc, fontsize = 15)

        if (show):
            plt.show()
            plt.close(fig)

    # NOTE Currently don't use the total mass, just the density mass specified (90%).
    return alpha_acceptances, alpha_range, alpha0, alpha1, alpha_mode, hdi_mass

# ==================================================================================

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
    
    # Case 1: alpha0 > 0.05, then imply SBS88 presence.
    if (a0 > 0.05):
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

def detect_sbs88(baseline_sig, healthy_profile, sbs88_profile, 
                 N_muts_range = np.array([])):
    """
    Give a baseline signature.
    Detect the presence of SBS88 assuming it has a range of mutations present.
    The estimated alpha parameter range is used to best tune the tolerance.

    Returns: the number of times it was detected per N_mut, and the corresponding alphas.
    """

    if not (N_muts_range.any()):
        N_muts_range = np.concat(([5, 10, 20], np.arange(50, 300 + 25, 25), [350, 400, 450, 500]))

    detected = np.zeros_like(N_muts_range, dtype = bool)
    estimated_alphas = np.zeros_like(N_muts_range, dtype = float)
    best_tol_matrix = helpers.retrieve_best_tol(N_muts_range)

    for j, N_muts in enumerate(N_muts_range):   
        best_tol_list = np.unique(best_tol_matrix[j])
        all_alphas = 0; alphas_detected = 0

        for best_tol in best_tol_list:
            # alpha_acceptances, alpha_range, alpha0, alpha1, alpha_mode, hdi_mass
            all_results = abc_alpha_inference(baseline_sig, healthy_profile, sbs88_profile,
                                                            N_muts, tol = best_tol)

            # Extract the HDI endpoints and use with the classifier.
            a0, a1, a_mode = all_results[2:5]
            sbs88_detected, estimated_alpha = abc_alpha_classifier(a0, a1, a_mode)
            if sbs88_detected:
                all_alphas += estimated_alpha 
                alphas_detected += 1
        
        avgd_alpha = all_alphas / alphas_detected if alphas_detected > 0 else 0
        detected[j] = (avgd_alpha > 0.05); estimated_alphas[j] = avgd_alpha 
    
    return detected, estimated_alphas, N_muts_range

# ====================================================================================

def test_classifier(healthy_profile, sbs88_profile, alpha_test, N_muts_test,
                     trials = 5, plot = False, save = False):
    """ 
    Generate a synthetic signature according to alpha_test and N_muts_test.
    See if the classifier recovers alpha/sbs88 at all.
    """

    # Store results in a list, convert to data frame at the end.
    test_results_list = []
    test_profile = synthetic_data.composite_profile(healthy_profile, sbs88_profile, alpha_test)

    for _ in range(trials):
        # NOTE Could look to modify detect_sbs88 with vectorisation.
        test_sig = synthetic_data.generate_signature(test_profile, N_muts_test)
        detected, estimated_alphas, compared_N_muts = detect_sbs88(test_sig, healthy_profile, sbs88_profile)

        for j in range(len(compared_N_muts)):
            test_results_list.append({
            # Store detected percentage rate.
            "detected": int(detected[j]) / trials,
            "est_alpha": estimated_alphas[j],
            "N_muts": compared_N_muts[j]
            })

    # After all trials, convert to data frame.
    test_df = pd.DataFrame(test_results_list)
    test_df = test_df.replace(0, np.nan) # So that the mean in aggregate works.

    summary = test_df.groupby("N_muts", as_index = False).agg({
        "detected": "sum",
        "est_alpha": "mean"
    })

    if (plot):
        plot_classifier_results(summary, alpha_test, N_muts_test, save = save)

    return summary

# ====================================================================================

def plot_classifier_results(summary_df, true_alpha = 0, true_N_mut = 0, title = None, save = False):
    """ 
    Plot the results of the classifier.
    """

    x = summary_df["N_muts"]
    y = summary_df["est_alpha"]

    if not (y.notna().any()):
        warnings.warn(f"No alpha found (all NaN), not plotting. True alpha = {true_alpha} with NM = {true_N_mut}.")
        return

    fig, ax = plt.subplots(figsize = (8, 4), tight_layout = True)

    for j in range(len(x)):
        conf = summary_df["detected"].iloc[j]
        ax.plot(x.iloc[j], y.iloc[j], marker = "o", color = "blue", 
                markersize = max(4, conf*8), alpha = max(0.15, conf))

    ax.plot(x, y, color = "blue", alpha = 0.5, label = "Classifier Estimates")
    if (true_alpha > 0):
        ax.axhline(y = true_alpha, linestyle = "dashed", color = "black", label = r"True $\alpha$")
    
    if (true_N_mut > 0):
        ax.axvline(x = true_N_mut, linestyle = "dashed", color = "black", label = r"True NM")

    ax.set_xlabel("Number of Mutations", fontsize = 12)
    ax.set_ylabel(r"Estimated Injection Level $\alpha$", fontsize = 12)
    ax.set_ylim(0.045, max(np.max(y) + 0.125, true_alpha + 0.025))

    if not title:
        title = r"Results of the ABC classifier against a synthetic mutational signature."

    ax.set_title(title, fontsize = 14)
    ax.grid(True, alpha = 0.5)
    ax.legend(loc = "upper right", fontsize = 13)

    if (save):
        randint = np.random.randint(0, 999)
        plt.savefig(f"classifier_results{randint}.png", format = "png", dpi = 200)

    plt.show()