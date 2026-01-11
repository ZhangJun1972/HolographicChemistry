"""
Core holographic chemistry framework.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from scipy import special, optimize, stats
import warnings

# Physical constants
FINE_STRUCTURE_CONSTANT = 1 / 137.036
BOHR_RADIUS = 0.529177  # Å
RYDBERG_ENERGY = 13.6057  # eV

@dataclass
class ElementProperties:
    """Data class for element properties."""
    Z: int
    symbol: str
    name: str
    period: int
    group: int
    block: str  # s, p, d, f
    is_noble: bool = False
    covalent_radius: Optional[float] = None
    uncertainty: Optional[float] = None
    electron_configuration: Optional[str] = None

class HolographicModel:
    """
    Base class for holographic models of covalent radii.
    
    Implements the holographic duality framework for atomic properties.
    """
    
    def __init__(self, alpha: float = FINE_STRUCTURE_CONSTANT):
        """
        Initialize holographic model.
        
        Parameters
        ----------
        alpha : float, optional
            Fine structure constant (default: 1/137.036)
        """
        self.alpha = alpha
        self.alpha2 = alpha ** 2
        self.alpha4 = alpha ** 4
        self.noble_gases = [2, 10, 18, 36, 54, 86]
        
        # Default parameters from Bayesian posterior means
        self.params = {
            'a_main': 1.52,
            'b_main': 0.285,
            'c_main': 15.1,
            'a_noble': 1.48,
            'b_noble': 0.272,
            'c_noble': 18.3,
            'd': 0.121,
            'beta': 0.0012,
            'epsilon': 850,
            'Z0': 36  # Kr reference
        }
        
        self._validate_parameters()
    
    def _validate_parameters(self):
        """Validate model parameters."""
        for key, value in self.params.items():
            if not isinstance(value, (int, float)):
                raise ValueError(f"Parameter {key} must be numeric")
            if key in ['a_main', 'a_noble', 'c_main', 'c_noble', 'd', 'epsilon']:
                if value < 0:
                    raise ValueError(f"Parameter {key} must be positive")
    
    def set_parameters(self, **kwargs):
        """Update model parameters."""
        for key, value in kwargs.items():
            if key in self.params:
                self.params[key] = value
            else:
                warnings.warn(f"Parameter {key} not recognized")
        self._validate_parameters()
    
    def is_noble_gas(self, Z: Union[int, np.ndarray]) -> Union[bool, np.ndarray]:
        """Check if element(s) is/are noble gas(es)."""
        if isinstance(Z, (int, np.integer)):
            return Z in self.noble_gases
        else:
            return np.isin(Z, self.noble_gases)
    
    def mean_field_term(self, Z: np.ndarray, is_noble: np.ndarray) -> np.ndarray:
        """Calculate mean field contribution (a * Z^(-b))."""
        a = np.where(is_noble, self.params['a_noble'], self.params['a_main'])
        b = np.where(is_noble, self.params['b_noble'], self.params['b_main'])
        return a * np.power(Z, -b)
    
    def qed_correction_term(self, Z: np.ndarray, is_noble: np.ndarray) -> np.ndarray:
        """Calculate QED/entanglement correction (c * α² * Z^(2/3))."""
        c = np.where(is_noble, self.params['c_noble'], self.params['c_main'])
        return c * self.alpha2 * np.power(Z, 2/3)
    
    def noble_gas_correction(self, Z: np.ndarray) -> np.ndarray:
        """Calculate noble gas specific correction."""
        if not np.any(self.is_noble_gas(Z)):
            return np.zeros_like(Z)
        
        # Gaussian term for relativistic maximum near Kr
        Z0 = self.params['Z0']
        gaussian = np.exp(-self.params['beta'] * (Z - Z0) ** 2)
        
        # Higher-order QED term
        higher_order = self.params['epsilon'] * self.alpha4 * np.power(Z, 4/3)
        
        # Apply only to noble gases
        noble_mask = self.is_noble_gas(Z)
        correction = np.zeros_like(Z)
        correction[noble_mask] = (
            self.params['d'] * gaussian[noble_mask] + 
            higher_order[noble_mask]
        )
        
        return correction
    
    def predict(self, Z: Union[int, np.ndarray], element_type: Optional[str] = None) -> Union[float, np.ndarray]:
        """
        Predict covalent radius for given atomic number(s).
        
        Parameters
        ----------
        Z : int or array-like
            Atomic number(s)
        element_type : str, optional
            'main' or 'noble', if None auto-detects from Z
            
        Returns
        -------
        float or ndarray
            Predicted covalent radius in Å
        """
        Z = np.asarray(Z)
        
        if element_type is None:
            is_noble = self.is_noble_gas(Z)
        else:
            if element_type == 'noble':
                is_noble = np.ones_like(Z, dtype=bool)
            elif element_type == 'main':
                is_noble = np.zeros_like(Z, dtype=bool)
            else:
                raise ValueError("element_type must be 'main' or 'noble'")
        
        # Calculate contributions
        mean_field = self.mean_field_term(Z, is_noble)
        qed_correction = self.qed_correction_term(Z, is_noble)
        noble_correction = self.noble_gas_correction(Z)
        
        # Total prediction
        prediction = mean_field + qed_correction + noble_correction
        
        # Return scalar for single input
        if prediction.size == 1:
            return float(prediction)
        return prediction
    
    def contribution_analysis(self, Z: Union[int, np.ndarray]) -> Dict[str, Union[float, np.ndarray]]:
        """
        Analyze contributions of each term.
        
        Parameters
        ----------
        Z : int or array-like
            Atomic number(s)
            
        Returns
        -------
        dict
            Dictionary with contribution percentages
        """
        Z = np.asarray(Z)
        is_noble = self.is_noble_gas(Z)
        
        mean_field = self.mean_field_term(Z, is_noble)
        qed_correction = self.qed_correction_term(Z, is_noble)
        noble_correction = self.noble_gas_correction(Z)
        
        total = mean_field + qed_correction + noble_correction
        
        return {
            'mean_field': mean_field,
            'qed_correction': qed_correction,
            'noble_correction': noble_correction,
            'total': total,
            'mean_field_pct': 100 * mean_field / total,
            'qed_correction_pct': 100 * qed_correction / total,
            'noble_correction_pct': 100 * noble_correction / total
        }
    
    def entanglement_entropy_proxy(self, Z: np.ndarray, method: str = 'orbital_occupation') -> np.ndarray:
        """
        Calculate entanglement entropy proxy.
        
        Methods:
        - 'orbital_occupation': Based on orbital filling
        - 'ionization_energy': Based on ionization energies
        - 'simple_approximation': Simple Z-based approximation
        """
        if method == 'orbital_occupation':
            # Simplified orbital entropy calculation
            S = np.zeros_like(Z, dtype=float)
            for i, z in enumerate(Z):
                if z <= 2:  # H, He
                    S[i] = 0.5 * z
                elif z <= 10:  # Li to Ne
                    S[i] = 1.0 + 0.8 * (z - 2)
                elif z <= 18:  # Na to Ar
                    S[i] = 3.0 + 0.7 * (z - 10)
                else:
                    S[i] = 5.0 + 0.5 * np.log(z)
            return S
            
        elif method == 'simple_approximation':
            # Simple Z-based approximation
            return 0.5 * np.power(Z, 0.75)
            
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def holographic_relation_test(self, Z: np.ndarray, r_experimental: np.ndarray) -> Dict:
        """
        Test holographic relation: r ∝ S_ent / G_N(Z)
        
        where G_N(Z) = 1 / (1 + κ α² Z^(2/3))
        """
        # Calculate entanglement entropy proxy
        S_ent = self.entanglement_entropy_proxy(Z)
        
        # Calculate effective gravitational constant
        kappa = 9.8  # Fitted parameter
        G_N = 1 / (1 + kappa * self.alpha2 * np.power(Z, 2/3))
        
        # Test proportionality
        lhs = r_experimental
        rhs = S_ent / G_N
        
        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(rhs, lhs)
        
        return {
            'correlation': r_value,
            'r_squared': r_value ** 2,
            'slope': slope,
            'intercept': intercept,
            'p_value': p_value,
            'std_err': std_err,
            'S_ent': S_ent,
            'G_N': G_N
        }
    
    def relativistic_contraction_factor(self, Z: int, n: int, l: int) -> float:
        """
        Calculate relativistic contraction factor for orbital.
        
        Parameters
        ----------
        Z : int
            Atomic number
        n : int
            Principal quantum number
        l : int
            Angular quantum number
            
        Returns
        -------
        float
            Contraction factor (<1 for contraction)
        """
        # Dirac equation relativistic factor
        gamma = np.sqrt(1 - (self.alpha * Z) ** 2)
        
        # Non-relativistic Bohr radius
        a0_nonrel = BOHR_RADIUS
        
        # Relativistic Bohr radius
        a0_rel = BOHR_RADIUS / gamma
        
        # Effective quantum number with quantum defect
        delta = self._quantum_defect(l, Z)
        n_eff = n - delta
        
        # Contraction factor
        contraction = (a0_rel * n_eff ** 2) / (a0_nonrel * n ** 2)
        
        return contraction
    
    def _quantum_defect(self, l: int, Z: int) -> float:
        """Calculate quantum defect including relativistic effects."""
        if l == 0:  # s orbitals
            return 0.1 + 0.001 * Z ** 1.5
        elif l == 1:  # p orbitals
            return 0.05 + 0.0005 * Z ** 1.5
        else:
            return 0.0

class NobleGasCorrection(HolographicModel):
    """
    Specialized model for noble gas corrections.
    
    Extends HolographicModel with additional noble gas physics.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Noble gas specific parameters
        self.topological_numbers = {
            2: 2,    # He
            10: 8,   # Ne
            18: 18,  # Ar
            36: 36,  # Kr
            54: 54,  # Xe
            86: 86   # Rn
        }
    
    def topological_invariant(self, Z: int) -> float:
        """Calculate topological invariant (Chern number analog)."""
        if Z not in self.topological_numbers:
            raise ValueError(f"No topological invariant for Z={Z}")
        return self.topological_numbers[Z]
    
    def shell_structure_factor(self, Z: int) -> float:
        """Calculate shell structure factor for noble gases."""
        if Z not in self.noble_gases:
            warnings.warn(f"Z={Z} is not a noble gas")
            return 1.0
        
        # Perfect closed-shell structure factor
        n_shells = self._count_shells(Z)
        return 1.0 + 0.05 * n_shells  # Small enhancement per shell
    
    def _count_shells(self, Z: int) -> int:
        """Count number of filled electron shells."""
        if Z <= 2: return 1
        elif Z <= 10: return 2
        elif Z <= 18: return 3
        elif Z <= 36: return 4
        elif Z <= 54: return 5
        elif Z <= 86: return 6
        else: return 7

class PeriodicTableAnalyzer:
    """
    Analyze periodic trends using holographic framework.
    """
    
    def __init__(self, model: HolographicModel):
        self.model = model
        
        # Periodic table structure
        self.periods = {
            1: list(range(1, 3)),
            2: list(range(3, 11)),
            3: list(range(11, 19)),
            4: list(range(19, 37)),
            5: list(range(37, 55)),
            6: list(range(55, 87)),
            7: list(range(87, 119))
        }
        
        self.groups = {
            1: [1, 3, 11, 19, 37, 55, 87],  # Alkali metals
            2: [4, 12, 20, 38, 56, 88],     # Alkaline earth
            13: [5, 13, 31, 49, 81, 113],   # Boron group
            14: [6, 14, 32, 50, 82, 114],   # Carbon group
            15: [7, 15, 33, 51, 83, 115],   # Nitrogen group
            16: [8, 16, 34, 52, 84, 116],   # Oxygen group
            17: [9, 17, 35, 53, 85, 117],   # Halogens
            18: [2, 10, 18, 36, 54, 86, 118] # Noble gases
        }
    
    def analyze_periodic_trends(self, Z_data: np.ndarray, r_data: np.ndarray) -> pd.DataFrame:
        """Analyze trends within each period."""
        results = []
        
        for period, Z_range in self.periods.items():
            mask = np.isin(Z_data, Z_range)
            if np.sum(mask) < 2:
                continue
            
            Z_period = Z_data[mask]
            r_period = r_data[mask]
            
            # Predictions
            r_pred = self.model.predict(Z_period)
            
            # Statistics
            residuals = r_period - r_pred
            mae = np.mean(np.abs(residuals))
            rmse = np.sqrt(np.mean(residuals ** 2))
            
            # Trend analysis
            if len(Z_period) > 1:
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    Z_period, r_period
                )
            else:
                slope = r_value = p_value = 0
            
            results.append({
                'period': period,
                'n_elements': len(Z_period),
                'mean_radius': np.mean(r_period),
                'radius_range': np.max(r_period) - np.min(r_period),
                'mae': mae,
                'rmse': rmse,
                'slope': slope,
                'r_squared': r_value ** 2,
                'p_value': p_value
            })
        
        return pd.DataFrame(results)
    
    def analyze_group_trends(self, Z_data: np.ndarray, r_data: np.ndarray) -> pd.DataFrame:
        """Analyze trends within each group."""
        results = []
        
        for group, Z_list in self.groups.items():
            mask = np.isin(Z_data, Z_list)
            if np.sum(mask) < 2:
                continue
            
            Z_group = Z_data[mask]
            r_group = r_data[mask]
            
            # Predictions
            r_pred = self.model.predict(Z_group)
            
            # Statistics
            residuals = r_group - r_pred
            mae = np.mean(np.abs(residuals))
            
            # Logarithmic scaling (expected for power laws)
            log_Z = np.log(Z_group)
            log_r = np.log(r_group)
            
            if len(Z_group) > 1:
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    log_Z, log_r
                )
                power_law_exponent = -slope
            else:
                power_law_exponent = r_value = p_value = 0
            
            results.append({
                'group': group,
                'n_elements': len(Z_group),
                'mean_radius': np.mean(r_group),
                'mae': mae,
                'power_law_exponent': power_law_exponent,
                'r_squared': r_value ** 2,
                'p_value': p_value
            })
        
        return pd.DataFrame(results)
    
    def discrete_scale_invariance_test(self, Z_data: np.ndarray, r_data: np.ndarray) -> Dict:
        """
        Test discrete scale invariance of the model.
        
        Calculates dimensionless combination r * Z^b / a
        and checks its constancy.
        """
        # Calculate dimensionless combination
        b_main = self.model.params['b_main']
        a_main = self.model.params['a_main']
        
        dimensionless = r_data * np.power(Z_data, b_main) / a_main
        
        # Statistics
        cv = np.std(dimensionless) / np.mean(dimensionless)  # Coefficient of variation
        
        # Compare with model predictions
        r_pred = self.model.predict(Z_data)
        dimensionless_pred = r_pred * np.power(Z_data, b_main) / a_main
        cv_pred = np.std(dimensionless_pred) / np.mean(dimensionless_pred)
        
        return {
            'dimensionless_values': dimensionless,
            'dimensionless_pred': dimensionless_pred,
            'cv_actual': cv,
            'cv_predicted': cv_pred,
            'improvement': (cv - cv_pred) / cv * 100,
            'mean_actual': np.mean(dimensionless),
            'mean_predicted': np.mean(dimensionless_pred)
      }
