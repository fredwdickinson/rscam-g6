from scipy.stats import mannwhitneyu

def perform_MWU(sample1, sample2, alpha, verbose = True):
    stat, p_value = mannwhitneyu(sample1, sample2)
    rejected = p_value < alpha

    if verbose:
        print('Statistics=%.2f, p=%.5f' % (stat, p_value))
        if rejected:
            print('Reject Null Hypothesis (Significant difference between two samples)')
        else:
            print('Do not Reject Null Hypothesis (No significant difference between two samples)')
    else:
        return rejected

def run_simulation(sample1, sample2, alpha =0.05, n_iterations = 100, sample_size = 35):
    rejections = 0
    for _ in range(n_iterations):
        rejection = perform_MWU(
            sample1.sample(n=sample_size)["Age"].tolist(),
            sample2.sample(n=sample_size)["Age"].tolist(),
            alpha,
            verbose = False
        )
        rejections += rejection

    print(f"Rejections: {rejections} in {n_iterations} tests")