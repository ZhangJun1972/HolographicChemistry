"""
Utility functions for holographic chemistry.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import warnings
from scipy import stats, special, optimize
import json
import yaml
import pickle
from pathlib import Path

def load_element_data(filepath: str) -> pd.DataFrame:
    """
    Load element data from CSV file.
    
    Expected columns:
    - Z: atomic number
    - symbol: element symbol
    - name: element name
    - period: period number
    - group: group number
    - covalent_radius: covalent radius in Å
    - uncertainty: measurement uncertainty
    - electron_configuration: electron configuration string
    """
    df = pd.read_csv(filepath)
    
    # Validate required columns
    required_columns = ['Z', 'symbol', 'period', 'group', 'covalent_radius']
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Set index
    df = df.set_index('Z').sort_index()
    
    return df

def save_results(results: Dict, filename: str, format: str = 'json'):
    """
    Save results to file.
    
    Parameters
    ----------
    results : dict
        Results dictionary
    filename : str
        Output filename
    format : str
        Output format: 'json', 'yaml', 'pkl'
    """
    filepath = Path(filename)
    
    if format == 'json':
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    
    elif format == 'yaml':
        with open(filepath, 'w') as f:
            yaml.dump(results, f, default_flow_style=False)
    
    elif format == 'pkl':
        with open(filepath, 'wb') as f:
            pickle.dump(results, f)
    
    else:
        raise ValueError(f"Unknown format: {format}")

def load_results(filename: str, format: str = 'json') -> Dict:
    """
    Load results from file.
    
    Parameters
    ----------
    filename : str
        Input filename
    format : str
        Input format: 'json', 'yaml', 'pkl'
    """
    filepath = Path(filename)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filename}")
    
    if format == 'json':
        with open(filepath, 'r') as f:
            return json.load(f)
    
    elif format == 'yaml':
        with open(filepath, 'r') as f:
            return yaml.safe_load(f)
    
    elif format == 'pkl':
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    else:
        raise ValueError(f"Unknown format: {format}")

def calculate_statistics(data: np.ndarray) -> Dict:
    """
    Calculate basic statistics for data array.
    """
    return {
        'mean': np.mean(data),
        'std': np.std(data),
        'median': np.median(data),
        'min': np.min(data),
        'max': np.max(data),
        'q1': np.percentile(data, 25),
        'q3': np.percentile(data, 75),
        'skewness': stats.skew(data),
        'kurtosis': stats.kurtosis(data),
        'n': len(data),
        'missing': np.sum(np.isnan(data))
    }

def weighted_statistics(values: np.ndarray, weights: np.ndarray) -> Dict:
    """
    Calculate weighted statistics.
    """
    if weights is None:
        weights = np.ones_like(values)
    
    # Normalize weights
    weights = weights / np.sum(weights)
    
    weighted_mean = np.sum(values * weights)
    weighted_var = np.sum(weights * (values - weighted_mean) ** 2)
    weighted_std = np.sqrt(weighted_var)
    
    return {
        'weighted_mean': weighted_mean,
        'weighted_std': weighted_std,
        'weighted_var': weighted_var,
        'effective_n': 1 / np.sum(weights ** 2)
    }

def confidence_interval(data: np.ndarray, confidence: float = 0.95) -> Tuple:
    """
    Calculate confidence interval.
    """
    n = len(data)
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    
    if n < 2:
        return (mean, mean)
    
    # t-distribution for small samples
    if n < 30:
        t = stats.t.ppf((1 + confidence) / 2, n - 1)
        margin = t * std / np.sqrt(n)
    else:
        # Normal approximation for large samples
        z = stats.norm.ppf((1 + confidence) / 2)
        margin = z * std / np.sqrt(n)
    
    return (mean - margin, mean + margin)

def bootstrap_confidence_interval(data: np.ndarray, statistic_func,
                                 n_bootstrap: int = 1000,
                                 confidence: float = 0.95) -> Dict:
    """
    Calculate bootstrap confidence interval.
    """
    n = len(data)
    bootstrap_stats = []
    
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        stat = statistic_func(sample)
        bootstrap_stats.append(stat)
    
    bootstrap_stats = np.array(bootstrap_stats)
    
    lower = np.percentile(bootstrap_stats, (1 - confidence) / 2 * 100)
    upper = np.percentile(bootstrap_stats, (1 + confidence) / 2 * 100)
    
    return {
        'statistic': statistic_func(data),
        'lower': lower,
        'upper': upper,
        'bootstrap_samples': bootstrap_stats,
        'bootstrap_mean': np.mean(bootstrap_stats),
        'bootstrap_std': np.std(bootstrap_stats)
    }

def correlation_matrix(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate correlation matrix with p-values.
    """
    corr_matrix = data.corr()
    
    # Calculate p-values
    n = len(data)
    p_matrix = pd.DataFrame(index=corr_matrix.index, columns=corr_matrix.columns)
    
    for i in corr_matrix.index:
        for j in corr_matrix.columns:
            if i == j:
                p_matrix.loc[i, j] = 0
            else:
                r = corr_matrix.loc[i, j]
                if np.isnan(r):
                    p_matrix.loc[i, j] = np.nan
                else:
                    t = r * np.sqrt((n - 2) / (1 - r ** 2))
                    p = 2 * (1 - stats.t.cdf(np.abs(t), n - 2))
                    p_matrix.loc[i, j] = p
    
    return corr_matrix, p_matrix

def format_scientific(value: float, precision: int = 3) -> str:
    """
    Format number in scientific notation.
    """
    if value == 0:
        return "0"
    
    exponent = np.floor(np.log10(np.abs(value)))
    mantissa = value / (10 ** exponent)
    
    if abs(exponent) <= 2:
        return f"{value:.{precision}f}"
    else:
        return f"{mantissa:.{precision}f} × 10^{int(exponent)}"

def element_symbol_to_Z(symbol: str) -> int:
    """
    Convert element symbol to atomic number.
    """
    # Basic periodic table mapping
    periodic_table = {
        'H': 1, 'He': 2,
        'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
        'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18,
        'K': 19, 'Ca': 20,
        'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27,
        'Ni': 28, 'Cu': 29, 'Zn': 30,
        'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36,
        'Rb': 37, 'Sr': 38,
        'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45,
        'Pd': 46, 'Ag': 47, 'Cd': 48,
        'In': 49, 'Sn': 50, 'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54,
        'Cs': 55, 'Ba': 56,
        'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60, 'Pm': 61, 'Sm': 62, 'Eu': 63,
        'Gd': 64, 'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70,
        'Lu': 71,
        'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78,
        'Au': 79, 'Hg': 80,
        'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86,
        'Fr': 87, 'Ra': 88,
        'Ac': 89, 'Th': 90, 'Pa': 91, 'U': 92, 'Np': 93, 'Pu': 94, 'Am': 95,
        'Cm': 96, 'Bk': 97, 'Cf': 98, 'Es': 99, 'Fm': 100, 'Md': 101,
        'No': 102, 'Lr': 103,
        'Rf': 104, 'Db': 105, 'Sg': 106, 'Bh': 107, 'Hs': 108, 'Mt': 109,
        'Ds': 110, 'Rg': 111, 'Cn': 112, 'Nh': 113, 'Fl': 114, 'Mc': 115,
        'Lv': 116, 'Ts': 117, 'Og': 118
    }
    
    symbol = symbol.capitalize()
    if symbol not in periodic_table:
        raise ValueError(f"Unknown element symbol: {symbol}")
    
    return periodic_table[symbol]

def Z_to_element_symbol(Z: int) -> str:
    """
    Convert atomic number to element symbol.
    """
    # Reverse mapping
    periodic_table = {v: k for k, v in element_symbol_to_Z.__closure__[0].cell_contents.items()}
    
    if Z not in periodic_table:
        raise ValueError(f"Unknown atomic number: {Z}")
    
    return periodic_table[Z]

def relativistic_correction(Z: int, method: str = 'dirac') -> float:
    """
    Calculate relativistic correction factor.
    
    Methods:
    - 'dirac': Dirac equation exact for hydrogen-like
    - 'screening': Screened nuclear charge approximation
    - 'empirical': Empirical fit
    """
    alpha = 1/137.036
    
    if method == 'dirac':
        # Exact Dirac solution for hydrogen-like
        gamma = np.sqrt(1 - (alpha * Z) ** 2)
        return 1 / gamma
    
    elif method == 'screening':
        # Screened nuclear charge
        Z_eff = Z - 0.3  # Simple screening
        gamma = np.sqrt(1 - (alpha * Z_eff) ** 2)
        return 1 / gamma
    
    elif method == 'empirical':
        # Empirical fit from literature
        return 1 + 0.006 * Z ** 1.5
    
    else:
        raise ValueError(f"Unknown method: {method}")

def quantum_defect(n: int, l: int, Z: int) -> float:
    """
    Calculate quantum defect for atomic orbitals.
    
    Parameters
    ----------
    n : int
        Principal quantum number
    l : int
        Angular quantum number
    Z : int
        Atomic number
    """
    # Simplified formula
    if l == 0:  # s orbitals
        return 0.41 + 0.00083 * Z ** 2
    elif l == 1:  # p orbitals
        return 0.04 - 0.00016 * Z ** 2
    elif l == 2:  # d orbitals
        return 0.00
    else:  # f and higher
        return 0.00

def compute_orbital_energy(n: int, l: int, Z: int) -> float:
    """
    Compute orbital energy in Rydberg units.
    """
    delta = quantum_defect(n, l, Z)
    n_eff = n - delta
    return -Z ** 2 / n_eff ** 2  # Rydberg units

def create_periodic_table_dataframe() -> pd.DataFrame:
    """
    Create comprehensive periodic table DataFrame.
    """
    data = []
    
    # Basic periodic table data
    elements = [
        # Z, symbol, name, period, group, block
        (1, 'H', 'Hydrogen', 1, 1, 's'),
        (2, 'He', 'Helium', 1, 18, 's'),
        (3, 'Li', 'Lithium', 2, 1, 's'),
        (4, 'Be', 'Beryllium', 2, 2, 's'),
        (5, 'B', 'Boron', 2, 13, 'p'),
        (6, 'C', 'Carbon', 2, 14, 'p'),
        (7, 'N', 'Nitrogen', 2, 15, 'p'),
        (8, 'O', 'Oxygen', 2, 16, 'p'),
        (9, 'F', 'Fluorine', 2, 17, 'p'),
        (10, 'Ne', 'Neon', 2, 18, 'p'),
        # ... continue for all elements
    ]
    
    for Z, symbol, name, period, group, block in elements:
        data.append({
            'Z': Z,
            'symbol': symbol,
            'name': name,
            'period': period,
            'group': group,
            'block': block,
            'is_noble': Z in [2, 10, 18, 36, 54, 86]
        })
    
    return pd.DataFrame(data)

def validate_model_parameters(params: Dict) -> bool:
    """
    Validate model parameters.
    
    Returns
    -------
    bool
        True if parameters are valid
    """
    required = ['a_main', 'b_main', 'c_main']
    
    # Check required parameters
    for key in required:
        if key not in params:
            raise ValueError(f"Missing required parameter: {key}")
    
    # Check parameter ranges
    if params['a_main'] <= 0:
        raise ValueError("a_main must be positive")
    
    if not 0 < params['b_main'] < 1:
        raise ValueError("b_main must be between 0 and 1")
    
    if params['c_main'] < 0:
        raise ValueError("c_main must be non-negative")
    
    # Check noble gas parameters if present
    if 'd' in params and params['d'] < 0:
        raise ValueError("d must be non-negative")
    
    if 'epsilon' in params and params['epsilon'] < 0:
        raise ValueError("epsilon must be non-negative")
    
    return True

def generate_parameter_report(params: Dict) -> str:
    """
    Generate formatted parameter report.
    """
    report = "Model Parameters Report\n"
    report += "=" * 50 + "\n\n"
    
    for key, value in sorted(params.items()):
        if isinstance(value, float):
            report += f"{key:20s}: {value:10.6f}\n"
        else:
            report += f"{key:20s}: {value}\n"
    
    report += "\n" + "=" * 50 + "\n"
    
    return report

def compute_goodness_of_fit(y_true: np.ndarray, y_pred: np.ndarray,
                           n_params: int) -> Dict:
    """
    Compute comprehensive goodness-of-fit statistics.
    """
    n = len(y_true)
    residuals = y_true - y_pred
    
    # Basic statistics
    r2 = 1 - np.sum(residuals ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - n_params - 1)
    
    # Information criteria
    rss = np.sum(residuals ** 2)
    aic = 2 * n_params + n * np.log(rss / n)
    bic = n_params * np.log(n) + n * np.log(rss / n)
    
    # Prediction error
    mse = np.mean(residuals ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(residuals))
    mape = np.mean(np.abs(residuals / y_true)) * 100
    
    # Statistical tests
    dw_stat = np.sum(np.diff(residuals) ** 2) / np.sum(residuals ** 2)
    
    return {
        'r2': r2,
        'adj_r2': adj_r2,
        'aic': aic,
        'bic': bic,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'durbin_watson': dw_stat,
        'n': n,
        'n_params': n_params,
        'rss': rss
    }
