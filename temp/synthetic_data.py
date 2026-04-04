import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def normalise(v):
    return v / np.sum(v)

# ==================================================================================

def generate_signature(profile, N_muts, B = 1):
    """ 
    Generates a mutational signature given a normalised probability distribution
    (i.e. the profile) and the number of mutations presents.
        Normalises the result.
    """

    if (B == 1):
        # Keep old logic for notebook code.
        unnormalised = np.random.multinomial(N_muts, profile)
        return unnormalised / np.sum(unnormalised)
    else:
        unnormalised = np.random.multinomial(N_muts, profile, size = B)
        return unnormalised / np.sum(unnormalised)

# ==================================================================================

def composite_profile(healthy_profile, sbs88_profile, alpha):
    """ 
    Generates the mutational profile (probability distribution) according to
    a certain alpha and number of mutations, assumes
        composite = alphaSBS88 + (1-alpha)healthyColon.
    """

    # Normalisation step.
    profile_ = alpha*sbs88_profile + (1-alpha)*healthy_profile
    profile = normalise(profile_)

    return profile

# ===================================================================================

def create_mutational_profiles(healthy_profile, sbs88_profile, alpha_range = [0, 0.0843, 0.1, 0.25, 0.5, 0.8, 1]):
    """ 
    Generates a dictionary of commmonly used mutational profiles.
    """

    mutational_profiles = {}
    for alpha in alpha_range:
        mutational_profiles[alpha] = composite_profile(healthy_profile, sbs88_profile, alpha)

    return mutational_profiles


# ==================================================================================

def plot_signature(contexts, mutations, profile, N_muts = None, alpha = None):
    """ 
    Plot a mutational signature. Default sets the axis off.
    """

    fig, ax = plt.subplots(figsize = (8, 4), tight_layout = True)

    # Construct df and palette for seaborn plotting.
    plot_data = pd.DataFrame({"Context": contexts, "Mutation": mutations, "Density": profile})
    mutation_palette = {"T>A": "blue", "T>C": "red", "T>G": "green",
                        "C>T": "orange", "C>G": "pink", "C>A": "black"}

    
    sns.barplot(data = plot_data, x = "Context", y = "Density", 
                hue = "Mutation", palette = mutation_palette, ax = ax)
    
    ax.legend(loc = "best", fontsize = 13)
    ax.set_xticks([])

    if (N_muts):
        title = rf"Mutational signature with {N_muts} mutations and $\alpha =${alpha}."
        ax.set_title(title, fontsize = 14)
    

    plt.show()
    





