import importlib
import abc_models
import synthetic_data
import helpers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def optimal_alpha_tol(healthy_profile, sbs88_profile, alpha, N_muts_range, 
                      tol_range = np.array([]), baseline_comparisons = 100, plot = False):
    """ 
    Provide the mutational profile for a healhy colon, an alpha, and the sbs88 profile.
    Finds the "optimal tolerance" for abc-alpha, where "optimal" minimises the 
        TPR - FPR : how often does it recover signatures derived from alpha, against those from a healthy colon?
    Averages across a number of baseline signatures, each with (1000 default) batches.

    Returns: the list of best tolerances and best J scores corresponding to the given range of the number of mutations.
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
        baseline_healthy_sigs = synthetic_data.generate_signature(healthy_profile, N_muts, B = baseline_comparisons)
        baseline_infected_sigs = synthetic_data.generate_signature(infected_profile, N_muts, B = baseline_comparisons)
        # baseline_healthy_sigs = np.array([synthetic_data.generate_signature(healthy_profile, N_muts) for _ in range(baseline_comparisons)])
        # baseline_infected_sigs = np.array([synthetic_data.generate_signature(infected_profile, N_muts) for _ in range(baseline_comparisons)])
        
        # compare_alpha() will give the cosine difference from two mutational signatures.
        # Want the distances for each value of alpha, for each baseline signature, across all B=1000 batches.
        a_range = np.linspace(0, 1, 100) # Uses a_bins = 100.
        distances_tp = np.zeros((len(a_range), baseline_comparisons, 1000))
        distances_fp = np.zeros((len(a_range), baseline_comparisons, 1000))

        # Create distance matrix.
        for i, alpha_abc in enumerate(a_range):
            infected_profile2 = synthetic_data.composite_profile(healthy_profile, sbs88_profile, alpha_abc)
            synthetic_comparisons_mat = synthetic_data.generate_signature(infected_profile2, N_muts, B = 1000)

            for k in range(baseline_comparisons):
                # Compare infected signature across a range of alphas.
                distances_tp[i, k, :] = helpers.batch_cosine_sim(baseline_infected_sigs[k], synthetic_comparisons_mat)
                distances_fp[i, k, :] = helpers.batch_cosine_sim(baseline_healthy_sigs[k], synthetic_comparisons_mat)
                
        # Shape of both: 50 x baseline_comparisons x B
        # Normally 50 x 100 x 1000. Each row corresponds to an alpha.
        # Now compare across a range of tolerances: not recompute every time.
        best_tol = 0; best_J = -1;
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

                if (total_tp_accepted > 5):
                    a0_tp, a1_tp, mode_tp, _ = helpers.hdi(tp_posteriors_norm[:, k], a_range)
                    tp_count += abc_models.abc_alpha_classifier(a0_tp, a1_tp, mode_tp)[0]
                
                if (total_fp_accepted > 5):
                    a0_fp, a1_fp, mode_fp, _ = helpers.hdi(fp_posteriors_norm[:, k], a_range)
                    fp_count += abc_models.abc_alpha_classifier(a0_fp, a1_fp, mode_fp)[0]
                else:
                    fp_count += 0

            # Best J = TPR - FPR, averaged across the number of baseline_comparisons.
            J_tol = (tp_count - fp_count) / baseline_comparisons

            if (J_tol >= best_J): # Change to >= so bigger tolerances are also accepted.
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
        ax.legend(lines, labs, loc = "lower right", bbox_to_anchor = (1, 0.175), fontsize = 13)
        ax.set_title(rf"Optimal tolerance against the number of mutations for $\alpha = ${alpha}", fontsize = 14)

        plt.savefig(f"optimal-alpha-{alpha}.png", format = "png", dpi = 200)
        plt.show()

    return best_tols, best_Js

# ==================================================================================