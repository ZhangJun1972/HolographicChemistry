"""
Different model implementations for comparison.
"""

import numpy as np
from typing import Dict, List, Tuple
from scipy import optimize, stats
from .core import HolographicModel, FINE_STRUCTURE_CONSTANT

class BenchmarkModel:
    """
    Traditional benchmark model: r = a * Z^(-b)
    """
    
    def __init__(self):
        self.params = {'a': 1.82, 'b': 0.31}
        self.name = "Benchmark Model"
    
    def predict(self, Z):
        """Predict radius using power law."""
        Z = np.asarray(Z)
        return self.params['a'] * np.power(Z, -self.params['b'])
    
    def fit(self, Z, r, weights=None):
        """Fit model to data."""
        def objective(params, Z, r, weights):
            a, b = params
            pred = a * np.power(Z, -b)
            if weights is None:
                return np.sum((r - pred) ** 2)
            else:
                return np.sum(weights * (r - pred) ** 2)
        
        # Initial guess
        p0 = [self.params['a'], self.params['b']]
        
        # Bounds
        bounds = [(0.5, 3.0), (0.1, 0.5)]
        
        # Optimization
        result = optimize.minimize(
            objective, p0, args=(Z, r, weights),
            bounds=bounds, method='L-BFGS-B'
        )
        
        if result.success:
            self.params['a'], self.params['b'] = result.x
        else:
            warnings.warn(f"Fit failed: {result.message}")
        
        return result

class EmergentModel(HolographicModel):
    """
    Emergent model with α² correction: r = aZ^(-b) + cα²Z^(2/3)
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.params.update({
            'a': 1.51,
            'b': 0.28,
            'c': 15.3
        })
        self.name = "Emergent Model"
    
    def predict(self, Z, element_type=None):
        """Predict radius for emergent model."""
        Z = np.asarray(Z)
        
        mean_field = self.params['a'] * np.power(Z, -self.params['b'])
        qed_correction = self.params['c'] * self.alpha2 * np.power(Z, 2/3)
        
        prediction = mean_field + qed_correction
        
        if prediction.size == 1:
            return float(prediction)
        return prediction
    
    def fit(self, Z, r, weights=None):
        """Fit emergent model to data."""
        def objective(params, Z, r, weights):
            a, b, c = params
            pred = a * np.power(Z, -b) + c * self.alpha2 * np.power(Z, 2/3)
            if weights is None:
                return np.sum((r - pred) ** 2)
            else:
                return np.sum(weights * (r - pred) ** 2)
        
        # Initial guess
        p0 = [self.params['a'], self.params['b'], self.params['c']]
        
        # Bounds
        bounds = [(0.5, 3.0), (0.1, 0.5), (0, 50)]
        
        # Optimization
        result = optimize.minimize(
            objective, p0, args=(Z, r, weights),
            bounds=bounds, method='L-BFGS-B'
        )
        
        if result.success:
            self.params['a'], self.params['b'], self.params['c'] = result.x
        else:
            warnings.warn(f"Fit failed: {result.message}")
        
        return result

class CompleteModel(HolographicModel):
    """
    Complete holographic model with noble gas corrections.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Complete Holographic Model"
    
    def fit_bayesian(self, Z, r, is_noble=None, **kwargs):
        """Fit using Bayesian methods (placeholder for actual implementation)."""
        from .bayesian import BayesianHierarchicalModel
        
        if is_noble is None:
            is_noble = self.is_noble_gas(Z)
        
        model = BayesianHierarchicalModel(Z, r, is_noble)
        trace = model.sample(**kwargs)
        
        # Update parameters with posterior means
        summary = model.summary(trace)
        for key in self.params:
            if key in summary['mean']:
                self.params[key] = summary['mean'][key]
        
        return trace, summary

class TransitionMetalExtension(HolographicModel):
    """
    Extension for transition metals with d/f orbital corrections.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Additional parameters for d/f electrons
        self.params.update({
            'eta_d': 0.05,  # d-electron coupling
            'eta_f': 0.08,  # f-electron coupling
            'zeta': 0.02    # magnetic correction
        })
        
        # Transition metal Z ranges
        self.d_block = list(range(21, 31)) + list(range(39, 49)) + list(range(72, 81))
        self.f_block = list(range(57, 72)) + list(range(89, 104))
    
    def d_electron_count(self, Z: int) -> int:
        """Count d electrons for given Z."""
        # Simplified counting
        if Z < 21:
            return 0
        elif 21 <= Z < 30:
            return Z - 20
        elif 30 <= Z < 39:
            return 10
        elif 39 <= Z < 48:
            return Z - 38
        elif 48 <= Z < 57:
            return 10
        elif 72 <= Z < 81:
            return Z - 71
        else:
            return 0
    
    def f_electron_count(self, Z: int) -> int:
        """Count f electrons for given Z."""
        if 57 <= Z < 72:
            return Z - 56
        elif 89 <= Z < 104:
            return Z - 88
        else:
            return 0
    
    def magnetic_moment(self, Z: int) -> float:
        """Estimate magnetic moment for transition metal."""
        d_count = self.d_electron_count(Z)
        if d_count <= 5:
            return d_count
        else:
            return 10 - d_count
    
    def transition_metal_correction(self, Z: np.ndarray) -> np.ndarray:
        """Calculate transition metal correction."""
        Z = np.asarray(Z)
        correction = np.zeros_like(Z, dtype=float)
        
        for i, z in enumerate(Z):
            if z in self.d_block or z in self.f_block:
                nd = self.d_electron_count(z)
                nf = self.f_electron_count(z)
                mu = self.magnetic_moment(z)
                
                d_corr = self.params['eta_d'] * nd * np.sqrt(z)
                f_corr = self.params['eta_f'] * nf * np.power(z, 0.75)
                mag_corr = self.params['zeta'] * mu * np.power(z, 1/3)
                
                correction[i] = d_corr + f_corr + mag_corr
        
        return correction
    
    def predict(self, Z, element_type=None):
        """Predict radius with transition metal corrections."""
        base_pred = super().predict(Z, element_type)
        
        if isinstance(Z, (int, np.integer)):
            Z = np.array([Z])
            scalar_output = True
        else:
            Z = np.asarray(Z)
            scalar_output = False
        
        tm_correction = self.transition_metal_correction(Z)
        prediction = base_pred + tm_correction
        
        if scalar_output:
            return float(prediction[0])
        return prediction
