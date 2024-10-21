import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy.stats import invwishart, multivariate_normal
import arviz as az


def sample_from_dataframe(data, sample_size, sp_factors, K=3, num_iterations=1000, burn_in=500):
    """
    Performs Gibbs sampling on the provided DataFrame and returns a subset of the DataFrame
    based on the sampling results.

    Parameters:
    - data: pandas DataFrame containing the trip data.
    - sample_size: int, the number of data points to sample.
    - sp_factors: list of columns for spatial and temporal factors.
    - K: int, number of clusters.
    - num_iterations: int, total number of Gibbs sampling iterations.
    - burn_in: int, number of iterations to discard as burn-in.

    Returns:
    - sampled_data: pandas DataFrame, subset of the original data.
    """

    X = data[sp_factors].values
    N, D = X.shape
    np.random.seed(0)    

    z = np.random.choice(K, size=N)    
    mu = np.zeros((K, D))
    Sigma = np.array([np.eye(D) for _ in range(K)])    
    mu_0 = X.mean(axis=0)
    lambda_param = 0.01
    nu_0 = D + 2
    Psi_0 = np.eye(D)
    
    mu_samples = []
    Sigma_samples = []
    z_samples = []
    
    for iter in range(num_iterations):
        # Step 1: Update cluster assignments z_i
        for i in range(N):
            log_probs = np.zeros(K)
            for k in range(K):
                try:
                    log_probs[k] = multivariate_normal.logpdf(
                        X[i], mean=mu[k], cov=Sigma[k], allow_singular=True
                    )
                except np.linalg.LinAlgError:
                    log_probs[k] = -np.inf  # Assign negligible probability
            # Normalize probabilities
            max_log_prob = np.max(log_probs)
            probs = np.exp(log_probs - max_log_prob)
            probs /= probs.sum()
            if np.isnan(probs).any():
                # Handle NaN probabilities
                probs = np.ones(K) / K
            z[i] = np.random.choice(K, p=probs)
        
        # Step 2: Update cluster parameters mu_k and Sigma_k
        for k in range(K):
            X_k = X[z == k]
            N_k = X_k.shape[0]
            if N_k > 0:
                x_bar_k = X_k.mean(axis=0)
                S_k = np.cov(X_k.T, bias=False) if N_k > 1 else np.zeros((D, D))
                lambda_n = lambda_param + N_k
                mu_n = (lambda_param * mu_0 + N_k * x_bar_k) / lambda_n
                nu_n = nu_0 + N_k
                diff = x_bar_k - mu_0
                Psi_n = Psi_0 + (N_k - 1) * S_k + \
                        (lambda_param * N_k / lambda_n) * np.outer(diff, diff)
                # Sample Sigma_k from inverse Wishart distribution
                try:
                    Sigma[k] = invwishart.rvs(df=nu_n, scale=Psi_n)
                except ValueError:
                    # In case of errors, use a small identity matrix
                    Sigma[k] = np.eye(D) * 1e-6
                # Ensure Sigma_k is positive definite
                if not np.all(np.linalg.eigvals(Sigma[k]) > 0):
                    Sigma[k] += np.eye(D) * 1e-6
                # Sample mu_k from multivariate normal distribution
                mu[k] = np.random.multivariate_normal(mu_n, Sigma[k] / lambda_n)
            else:
                # Re-initialize if no data assigned to cluster k
                Sigma[k] = np.eye(D)
                mu[k] = np.random.multivariate_normal(mu_0, Sigma[k])
        
        # Store samples after burn-in
        if iter >= burn_in:
            mu_samples.append(mu.copy())
            Sigma_samples.append(Sigma.copy())
            z_samples.append(z.copy())
    
    # After sampling, compute the posterior probabilities for each data point
    # We will use the last set of cluster assignments for simplicity
    final_z = z_samples[-1]
    
    # Compute the frequency of each data point being assigned to each cluster
    cluster_counts = np.zeros((N, K))
    for sample_z in z_samples:
        for i in range(N):
            cluster_counts[i, int(sample_z[i])] += 1
    # Normalize to get probabilities
    cluster_probs = cluster_counts / len(z_samples)
    
    # Compute overall probability of each data point (summing over clusters)
    data_point_probs = cluster_probs.sum(axis=1)
    
    # Normalize probabilities to sum to 1
    data_point_probs /= data_point_probs.sum()
    
    # Sample data points based on these probabilities
    sampled_indices = np.random.choice(N, size=sample_size, replace=False, p=data_point_probs)
    sampled_data = data.iloc[sampled_indices]
    
    return sampled_data


def sample_from_dataframe_uniform(data, sample_size):
    return data.sample(n=sample_size, random_state=0)


def select_number_of_clusters(data, sp_factors, k_min=1, k_max=10):
    """
    - data: pandas DataFrame containing the trip data.
    - k_min: int, minimum number of clusters to evaluate.
    - k_max: int, maximum number of clusters to evaluate.
    - optimal_k: int, the number of clusters with the lowest BIC.
    - bic_values: list of BIC values for each K.
    - k_values: list of K values evaluated.
    """
    X = data[sp_factors].values    
    bic_values = []
    k_values = range(k_min, k_max + 1)

    for K in k_values:
        gmm = GaussianMixture(n_components=K, covariance_type='full', random_state=0)
        gmm.fit(X)
        bic = gmm.bic(X)
        bic_values.append(bic)
        print(f'K={K}, BIC={bic}')
    
    optimal_k = k_values[np.argmin(bic_values)]
    print(f'\nOptimal number of clusters: {optimal_k}')
    plt.figure(figsize=(8, 6))
    plt.plot(k_values, bic_values, marker='o')
    plt.xlabel('Number of clusters (K)')
    plt.ylabel('BIC')
    plt.title('BIC vs. Number of Clusters')
    plt.show()
    
    return optimal_k, bic_values, k_values


def assess_convergence(mu_samples, parameter_names=None):
    """
    - mu_samples: list of numpy arrays, samples of mu parameters collected after burn-in.
                  Each element in the list corresponds to a Gibbs sampling iteration.
    - parameter_names: list of strings, names of the parameters for plotting. If None,
                       default names will be assigned.

    Returns:
    - convergence_metrics: dict containing ESS and PSRF for each parameter.
    """
    num_iterations = len(mu_samples)
    K, D = mu_samples[0].shape  # Number of clusters and dimensions
    mu_samples_array = np.array(mu_samples)  # Shape: (iterations, K, D)
    if parameter_names is None:
        parameter_names = [f'mu_{k}_{d}' for k in range(K) for d in range(D)]
    param_samples = {}
    for idx, name in enumerate(parameter_names):
        k = idx // D
        d = idx % D
        param_samples[name] = mu_samples_array[:, k, d]

    # trace plots
    fig, axes = plt.subplots(len(param_samples), figsize=(12, 2 * len(param_samples)), sharex=True)
    if len(param_samples) == 1:
        axes = [axes]  # Ensure axes is iterable
    for idx, (name, samples) in enumerate(param_samples.items()):
        axes[idx].plot(samples)
        axes[idx].set_title(f'Trace plot of {name}')
    plt.xlabel('Iteration')
    plt.tight_layout()
    plt.show()

    # autocorrelation
    fig, axes = plt.subplots(len(param_samples), figsize=(12, 2 * len(param_samples)), sharex=True)
    if len(param_samples) == 1:
        axes = [axes]
    for idx, (name, samples) in enumerate(param_samples.items()):
        az.plot_autocorr(samples, ax=axes[idx])
        axes[idx].set_title(f'Autocorrelation of {name}')
    plt.tight_layout()
    plt.show()

    inference_data = az.convert_to_inference_data(param_samples)
    ess = az.ess_trace(inference_data)
    rhat = az.rhat(inference_data)

    # convergence metrics
    convergence_metrics = {}
    print("Effective Sample Size (ESS):")
    for name in parameter_names:
        ess_value = ess.sel(var_names=name).values.item()
        convergence_metrics[f'ESS_{name}'] = ess_value
        print(f'{name}: {ess_value:.2f}')

    print("\nPotential Scale Reduction Factor (R-hat):")
    for name in parameter_names:
        rhat_value = rhat.sel(var_names=name).values.item()
        convergence_metrics[f'Rhat_{name}'] = rhat_value
        print(f'{name}: {rhat_value:.3f}')

    return convergence_metrics