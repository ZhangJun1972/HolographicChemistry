"""
Bayesian hierarchical modeling for parameter estimation.
"""

import numpy as np
import pandas as pd
import arviz as az
import pymc3 as pm
import theano.tensor as tt
from typing import Dict, List, Tuple, Optional
import warnings

class BayesianHierarchicalModel:
    """
    Bayesian hierarchical model for holographic chemistry.
    """
    
    def __init__(self, Z: np.ndarray, rcov: np.ndarray, groups: Dict):
        """
        Initialize Bayesian model.
        
        Parameters
        ----------
        Z : array-like
            Atomic numbers
        rcov : array-like
            Covalent radii
        groups : dict
            Grouping information with keys:
            - 'period': period number for each element
            - 'family': chemical family for each element
            - 'is_noble': boolean for noble gases
        """
        self.Z = np.asarray(Z)
        self.rcov = np.asarray(rcov)
        self.groups = groups
        
        # Constants
        self.alpha = 1/137.036
        self.alpha2 = self.alpha ** 2
        self.alpha4 = self.alpha ** 4
        
        # Preprocess groups
        self._process_groups()
        
        # Model storage
        self.model = None
        self.trace = None
    
    def _process_groups(self):
        """Process grouping information."""
        # Create period indices
        unique_periods = np.unique(self.groups['period'])
        self.period_to_idx = {p: i for i, p in enumerate(unique_periods)}
        self.period_idx = np.array([self.period_to_idx[p] for p in self.groups['period']])
        self.n_periods = len(unique_periods)
        
        # Create family indices
        unique_families = np.unique(self.groups['family'])
        self.family_to_idx = {f: i for i, f in enumerate(unique_families)}
        self.family_idx = np.array([self.family_to_idx[f] for f in self.groups['family']])
        self.n_families = len(unique_families)
        
        # Noble gas indicator
        self.is_noble = np.asarray(self.groups.get('is_noble', np.zeros_like(self.Z, dtype=bool)))
    
    def build_complete_model(self):
        """Build complete hierarchical model."""
        with pm.Model() as model:
            # Hyperpriors for main group elements
            mu_a_main = pm.Normal('mu_a_main', mu=1.5, sigma=0.3)
            sigma_a_main = pm.HalfNormal('sigma_a_main', sigma=0.2)
            
            mu_b_main = pm.Normal('mu_b_main', mu=0.3, sigma=0.1)
            sigma_b_main = pm.HalfNormal('sigma_b_main', sigma=0.05)
            
            mu_c_main = pm.Gamma('mu_c_main', alpha=2, beta=0.2)
            
            # Hyperpriors for noble gases
            mu_a_noble = pm.Normal('mu_a_noble', mu=mu_a_main, sigma=0.1)
            mu_b_noble = pm.Normal('mu_b_noble', mu=mu_b_main, sigma=0.05)
            mu_c_noble = pm.Gamma('mu_c_noble', alpha=3, beta=0.15)
            
            # Noble gas correction parameters
            d = pm.Normal('d', mu=0.1, sigma=0.05)
            beta = pm.Gamma('beta', alpha=1, beta=1000)
            epsilon = pm.Gamma('epsilon', alpha=2, beta=0.002)
            
            # Period-level parameters
            a_period = pm.Normal('a_period', 
                                mu=pm.math.switch(self.period_is_noble, mu_a_noble, mu_a_main),
                                sigma=sigma_a_main,
                                shape=self.n_periods)
            
            b_period = pm.Normal('b_period',
                                mu=pm.math.switch(self.period_is_noble, mu_b_noble, mu_b_main),
                                sigma=sigma_b_main,
                                shape=self.n_periods)
            
            c_period = pm.Gamma('c_period',
                               mu=pm.math.switch(self.period_is_noble, mu_c_noble, mu_c_main),
                               sigma=5.0,
                               shape=self.n_periods)
            
            # Individual parameters
            a = a_period[self.period_idx]
            b = b_period[self.period_idx]
            c = c_period[self.period_idx]
            
            # Model prediction
            Z_tensor = pm.constant(self.Z)
            is_noble_tensor = pm.constant(self.is_noble.astype(float))
            
            # Base terms
            mean_field = a * tt.pow(Z_tensor, -b)
            qed_correction = c * self.alpha2 * tt.pow(Z_tensor, 2/3)
            
            # Noble gas correction
            Z0 = 36
            gaussian = tt.exp(-beta * tt.pow(Z_tensor - Z0, 2))
            higher_order = epsilon * self.alpha4 * tt.pow(Z_tensor, 4/3)
            noble_correction = is_noble_tensor * (d * gaussian + higher_order)
            
            # Total prediction
            r_pred = mean_field + qed_correction + noble_correction
            
            # Observation error (higher for noble gases)
            sigma_obs_base = pm.HalfNormal('sigma_obs_base', sigma=0.03)
            sigma_obs_noble = pm.HalfNormal('sigma_obs_noble', sigma=0.05)
            sigma_obs = pm.math.switch(is_noble_tensor, sigma_obs_noble, sigma_obs_base)
            
            # Likelihood
            likelihood = pm.Normal('likelihood', mu=r_pred, sigma=sigma_obs, observed=self.rcov)
        
        self.model = model
        return model
    
    def sample(self, n_samples: int = 5000, tune: int = 2000, 
               chains: int = 4, **kwargs) -> az.InferenceData:
        """Sample from posterior distribution."""
        if self.model is None:
            self.build_complete_model()
        
        with self.model:
            self.trace = pm.sample(
                draws=n_samples,
                tune=tune,
                chains=chains,
                target_accept=0.9,
                return_inferencedata=True,
                **kwargs
            )
        
        return self.trace
    
    def summary(self, trace: Optional[az.InferenceData] = None) -> pd.DataFrame:
        """Generate summary statistics from trace."""
        if trace is None:
            if self.trace is None:
                raise ValueError("No trace available. Run sample() first.")
            trace = self.trace
        
        summary = az.summary(trace, round_to=4, hdi_prob=0.95)
        return summary
    
    def posterior_predictive_check(self, trace: Optional[az.InferenceData] = None, 
                                  n_samples: int = 1000) -> Dict:
        """Perform posterior predictive check."""
        if trace is None:
            if self.trace is None:
                raise ValueError("No trace available. Run sample() first.")
            trace = self.trace
        
        with self.model:
            ppc = pm.sample_posterior_predictive(trace, samples=n_samples, random_seed=42)
        
        # Calculate statistics
        ppc_mean = ppc['likelihood'].mean(axis=0)
        ppc_std = ppc['likelihood'].std(axis=0)
        
        # Calculate coverage
        lower = np.percentile(ppc['likelihood'], 2.5, axis=0)
        upper = np.percentile(ppc['likelihood'], 97.5, axis=0)
        coverage = np.mean((self.rcov >= lower) & (self.rcov <= upper))
        
        return {
            'ppc_samples': ppc['likelihood'],
            'ppc_mean': ppc_mean,
            'ppc_std': ppc_std,
            'coverage_95': coverage,
            'lower_bounds': lower,
            'upper_bounds': upper
        }
    
    def waic_loo(self, trace: Optional[az.InferenceData] = None) -> Dict:
        """Calculate WAIC and LOO for model comparison."""
        if trace is None:
            if self.trace is None:
                raise ValueError("No trace available. Run sample() first.")
            trace = self.trace
        
        waic_result = az.waic(trace)
        loo_result = az.loo(trace)
        
        return {
            'waic': waic_result.waic,
            'waic_se': waic_result.se,
            'p_waic': waic_result.p_waic,
            'loo': loo_result.loo,
            'loo_se': loo_result.se,
            'p_loo': loo_result.p_loo
        }

class MCMCSampler:
    """Wrapper for different MCMC sampling methods."""
    
    def __init__(self, model_type: str = 'complete'):
        self.model_type = model_type
        self.samplers = {
            'nuts': self._sample_nuts,
            'metropolis': self._sample_metropolis,
            'slice': self._sample_slice
        }
    
    def sample(self, model, method: str = 'nuts', **kwargs):
        """Sample using specified method."""
        if method not in self.samplers:
            raise ValueError(f"Unknown method: {method}. Available: {list(self.samplers.keys())}")
        
        return self.samplers[method](model, **kwargs)
    
    def _sample_nuts(self, model, **kwargs):
        """Sample using NUTS algorithm."""
        with model:
            trace = pm.sample(
                draws=kwargs.get('draws', 2000),
                tune=kwargs.get('tune', 1000),
                chains=kwargs.get('chains', 4),
                target_accept=kwargs.get('target_accept', 0.9),
                nuts={'max_treedepth': kwargs.get('max_treedepth', 15)},
                return_inferencedata=True
            )
        return trace
    
    def _sample_metropolis(self, model, **kwargs):
        """Sample using Metropolis-Hastings."""
        with model:
            step = pm.Metropolis()
            trace = pm.sample(
                draws=kwargs.get('draws', 5000),
                tune=kwargs.get('tune', 2000),
                chains=kwargs.get('chains', 4),
                step=step,
                return_inferencedata=True
            )
        return trace

class PosteriorAnalyzer:
    """Analyze posterior distributions."""
    
    def __init__(self, trace):
        self.trace = trace
        self.summary = az.summary(trace)
    
    def parameter_correlations(self) -> pd.DataFrame:
        """Calculate parameter correlations."""
        posterior = self.trace.posterior
        param_names = list(posterior.data_vars.keys())
        
        corr_matrix = np.zeros((len(param_names), len(param_names)))
        for i, name_i in enumerate(param_names):
            for j, name_j in enumerate(param_names):
                if i <= j:
                    data_i = posterior[name_i].values.flatten()
                    data_j = posterior[name_j].values.flatten()
                    corr = np.corrcoef(data_i, data_j)[0, 1]
                    corr_matrix[i, j] = corr_matrix[j, i] = corr
        
        return pd.DataFrame(corr_matrix, index=param_names, columns=param_names)
    
    def effective_sample_size(self) -> pd.DataFrame:
        """Calculate effective sample size."""
        ess = az.ess(self.trace)
        return ess
    
    def rhat_statistics(self) -> pd.DataFrame:
        """Calculate R-hat statistics."""
        rhat = az.rhat(self.trace)
        return rhat
    
    def posterior_predictions(self, Z: np.ndarray, n_samples: int = 1000) -> Dict:
        """Generate posterior predictions for new data."""
        # Extract parameter samples
        posterior = self.trace.posterior
        n_chains = posterior.chain.size
        n_draws = posterior.draw.size
        
        # Randomly select samples
        chain_idx = np.random.randint(0, n_chains, n_samples)
        draw_idx = np.random.randint(0, n_draws, n_samples)
        
        predictions = []
        for i in range(n_samples):
            # Get parameter values for this sample
            params = {}
            for var in posterior.data_vars:
                params[var] = float(posterior[var][chain_idx[i], draw_idx[i]])
            
            # Make prediction (simplified - needs actual model)
            # This would need to be adapted based on the actual model structure
            pred = self._predict_with_params(Z, params)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        return {
            'predictions': predictions,
            'mean': np.mean(predictions, axis=0),
            'std': np.std(predictions, axis=0),
            'lower_95': np.percentile(predictions, 2.5, axis=0),
            'upper_95': np.percentile(predictions, 97.5, axis=0)
        }
    
    def _predict_with_params(self, Z: np.ndarray, params: Dict) -> np.ndarray:
        """Predict with given parameters."""
        # Simplified prediction function
        alpha = 1/137.036
        alpha2 = alpha ** 2
        alpha4 = alpha ** 4
        
        mean_field = params.get('a_main', 1.52) * np.power(Z, -params.get('b_main', 0.285))
        qed_correction = params.get('c_main', 15.1) * alpha2 * np.power(Z, 2/3)
        
        # Noble gas correction
        is_noble = np.isin(Z, [2, 10, 18, 36, 54, 86])
        noble_corr = np.zeros_like(Z)
        if np.any(is_noble):
            Z0 = 36
            gaussian = np.exp(-params.get('beta', 0.0012) * (Z - Z0) ** 2)
            higher_order = params.get('epsilon', 850) * alpha4 * np.power(Z, 4/3)
            noble_corr[is_noble] = params.get('d', 0.121) * gaussian[is_noble] + higher_order[is_noble]
        
        return mean_field + qed_correction + noble_corr
