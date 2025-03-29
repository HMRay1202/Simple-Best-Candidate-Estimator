# Bayesian Optimal Stopping Simulation with Bootstrap Sampling

This project implements a simulation of an optimal stopping problem using a **bootstrap-based approximation to a Bayesian posterior**. The simulator is designed to determine, in a sequential candidate selection process, when to stop and select the current candidate based on predicted utility.

The core idea is to use bootstrap resampling to simulate uncertainty in the parameters of a linear regression model, forming an approximate posterior distribution. The model then uses this predictive distribution to decide whether a candidate is good enough to stop.

# File Structure

- `bootstrap_utils.py`  
  Module that contains the `generate_bootstrap_sample()` function, which reads a CSV file of candidate features and returns a bootstrap sample.

- `bayes_optimal_stopping.py`  
  Contains the `BayesianOptimalStopping` class and the `simulate_multiple()` function. This module implements the simulation logic using bootstrap-based posterior approximation and stopping decisions.

- `main.py`  
  Main script that ties everything together:
  - Loads input CSV data
  - Generates training samples via bootstrap
  - Constructs a true weight vector and response values
  - Runs multiple simulations
  - Outputs and saves results
 

# Installation

Make sure you have the following Python packages installed:

```bash
pip install numpy pandas scikit-learn tqdm
```

# Simulation Logic


1. Read candidate features from input CSV
2. Generate bootstrap samples from this dataset
3. Dynamically create a true weight vector `true_w` matching feature dimensions
4. Simulate utility scores using a linear model:  
    $y = X \cdot w_{true} + \varepsilon$
5. Use the `BayesianOptimalStopping` model to:
   - Fit many linear models via bootstrap to simulate posterior samples
   - Generate new candidates (based on observed data + noise)
   - Predict utility of each candidate using all bootstrapped models
   - Compute the probability that utility exceeds the current best
   - If that `probability` ≥ `decision_threshold`, stop and select

6. Repeat the above process for many runs (e.g. 1000), and report average performance

# Configurable Parameters

You can modify the following in `main.py`:

| Parameter | Description | Example |
|----------|-------------|---------|
| `decision_threshold` | Minimum probability to stop and select | 0.6 |
| `max_rounds` | Max rounds per simulation | 30 |
| `num_simulations` | Number of simulations | 1000 |
| `true_w` | True weight vector (can be fixed or randomly generated) | `np.random.uniform(...)` |
| `noise_std` | Standard deviation of utility noise | 0.1 |

# Theoretical Background

The model is based on Bayesian inference principles, using bootstrap sampling as a nonparametric approximation of the posterior distribution. This falls under:

- Empirical Bayes Approximation
- Non-parametric Posterior Simulation via Bootstrap

Decisions are based on the posterior predictive probability:

$\mathbb{P}(y_{\text{candidate}} > y^* \mid \text{data})$

If this probability exceeds the decision threshold, the candidate is selected.

# Future Improvements

- Support user upload of candidate CSV via interface
- Allow user-defined or fixed true weight vector
- Enable parameter sweep for simulation scenarios
- Visualize results (e.g., stopping round histogram, utility distributions)
- Add frontend interface (e.g., Streamlit or Flask)
- Implement true Bayesian inference via PyMC or Stan
