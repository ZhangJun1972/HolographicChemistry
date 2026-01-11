"""
Validation framework for holographic chemistry models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from scipy import stats, optimize
import warnings
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold, LeaveOneGroupOut
import matplotlib.pyplot as plt
import seaborn as sns

class ValidationMetrics:
    """
    Comprehensive validation metrics for model evaluation.
    """
    
    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray, 
                 n_params: int, model_name: str = ""):
        """
        Initialize validation metrics.
        
        Parameters
        ----------
        y_true : array-like
            True values
        y_pred : array-like
            Predicted values
        n_params : int
            Number of model parameters
        model_name : str
            Name of the model
        """
        self.y_true = np.asarray(y_true)
        self.y_pred = np.asarray(y_pred)
        self.n_params = n_params
        self.model_name = model_name
        self.n_samples = len(y_true)
        
        # Calculate residuals
        self.residuals = self.y_true - self.y_pred
        self.absolute_residuals = np.abs(self.residuals)
        self.relative_residuals = self.residuals / self.y_true * 100
    
    def calculate_all_metrics(self) -> Dict:
        """Calculate all validation metrics."""
        metrics = {}
        
        # Basic metrics
        metrics['r2'] = self.r_squared()
        metrics['adj_r2'] = self.adjusted_r_squared()
        metrics['mse'] = self.mse()
        metrics['rmse'] = self.rmse()
        metrics['mae'] = self.mae()
        metrics['mape'] = self.mape()
        metrics['max_ae'] = self.max_absolute_error()
        metrics['max_re'] = self.max_relative_error()
        
        # Information criteria
        metrics['aic'] = self.aic()
        metrics['bic'] = self.bic()
        
        # Statistical tests
        metrics['durbin_watson'] = self.durbin_watson()
        metrics['shapiro_p'] = self.shapiro_wilk_test()
        metrics['breusch_pagan'] = self.breusch_pagan_test()
        
        # Distribution statistics
        metrics['mean_residual'] = np.mean(self.residuals)
        metrics['std_residual'] = np.std(self.residuals)
        metrics['skew_residual'] = stats.skew(self.residuals)
        metrics['kurtosis_residual'] = stats.kurtosis(self.residuals)
        
        return metrics
    
    def r_squared(self) -> float:
        """Calculate R²."""
        return r2_score(self.y_true, self.y_pred)
    
    def adjusted_r_squared(self) -> float:
        """Calculate adjusted R²."""
        r2 = self.r_squared()
        return 1 - (1 - r2) * (self.n_samples - 1) / (self.n_samples - self.n_params - 1)
    
    def mse(self) -> float:
        """Calculate mean squared error."""
        return mean_squared_error(self.y_true, self.y_pred)
    
    def rmse(self) -> float:
        """Calculate root mean squared error."""
        return np.sqrt(self.mse())
    
    def mae(self) -> float:
        """Calculate mean absolute error."""
        return mean_absolute_error(self.y_true, self.y_pred)
    
    def mape(self) -> float:
        """Calculate mean absolute percentage error."""
        return np.mean(np.abs(self.relative_residuals))
    
    def max_absolute_error(self) -> float:
        """Calculate maximum absolute error."""
        return np.max(self.absolute_residuals)
    
    def max_relative_error(self) -> float:
        """Calculate maximum relative error."""
        return np.max(np.abs(self.relative_residuals))
    
    def aic(self) -> float:
        """Calculate Akaike Information Criterion."""
        rss = np.sum(self.residuals ** 2)
        return 2 * self.n_params + self.n_samples * np.log(rss / self.n_samples)
    
    def bic(self) -> float:
        """Calculate Bayesian Information Criterion."""
        rss = np.sum(self.residuals ** 2)
        return self.n_params * np.log(self.n_samples) + self.n_samples * np.log(rss / self.n_samples)
    
    def durbin_watson(self) -> float:
        """Calculate Durbin-Watson statistic for autocorrelation."""
        diff = np.diff(self.residuals)
        dw = np.sum(diff ** 2) / np.sum(self.residuals ** 2)
        return dw
    
    def shapiro_wilk_test(self) -> float:
        """Shapiro-Wilk test for normality of residuals."""
        if len(self.residuals) < 3:
            return np.nan
        _, p_value = stats.shapiro(self.residuals)
        return p_value
    
    def breusch_pagan_test(self) -> float:
        """Breusch-Pagan test for heteroscedasticity."""
        # Simple implementation
        squared_residuals = self.residuals ** 2
        _, p_value, _, _ = stats.ols(
            'squared_residuals ~ y_pred',
            data=pd.DataFrame({
                'squared_residuals': squared_residuals,
                'y_pred': self.y_pred
            })
        ).fit().f_test('y_pred = 0')
        return float(p_value)
    
    def summary_table(self) -> pd.DataFrame:
        """Generate summary table of metrics."""
        metrics = self.calculate_all_metrics()
        df = pd.DataFrame.from_dict(metrics, orient='index', columns=['Value'])
        df.index.name = 'Metric'
        return df.round(4)
    
    def compare_models(self, other_metrics: 'ValidationMetrics') -> pd.DataFrame:
        """Compare metrics with another model."""
        self_metrics = self.calculate_all_metrics()
        other_metrics_dict = other_metrics.calculate_all_metrics()
        
        comparison = pd.DataFrame({
            f'{self.model_name}': self_metrics,
            f'{other_metrics.model_name}': other_metrics_dict,
            'Difference': {k: self_metrics[k] - other_metrics_dict[k] 
                          for k in self_metrics.keys()},
            'Improvement %': {k: (other_metrics_dict[k] - self_metrics[k]) / 
                            abs(other_metrics_dict[k]) * 100 
                            if other_metrics_dict[k] != 0 else np.nan
                            for k in self_metrics.keys()}
        })
        
        return comparison

class CrossValidator:
    """
    Cross-validation framework for model evaluation.
    """
    
    def __init__(self, model, cv_method: str = 'kfold', n_splits: int = 5):
        """
        Initialize cross-validator.
        
        Parameters
        ----------
        model : object
            Model with fit() and predict() methods
        cv_method : str
            Cross-validation method: 'kfold', 'loo', 'logocv' (leave-one-group-out)
        n_splits : int
            Number of splits for KFold
        """
        self.model = model
        self.cv_method = cv_method
        self.n_splits = n_splits
        self.cv_results = []
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, 
                      groups: Optional[np.ndarray] = None,
                      weights: Optional[np.ndarray] = None) -> Dict:
        """
        Perform cross-validation.
        
        Parameters
        ----------
        X : array-like
            Features (atomic numbers)
        y : array-like
            Target values (covalent radii)
        groups : array-like, optional
            Group labels for group-based CV
        weights : array-like, optional
            Sample weights
            
        Returns
        -------
        dict
            Cross-validation results
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        if self.cv_method == 'kfold':
            return self._kfold_cv(X, y, weights)
        elif self.cv_method == 'loo':
            return self._leave_one_out_cv(X, y, weights)
        elif self.cv_method == 'logocv':
            if groups is None:
                raise ValueError("groups required for leave-one-group-out CV")
            return self._leave_one_group_out_cv(X, y, groups, weights)
        else:
            raise ValueError(f"Unknown CV method: {self.cv_method}")
    
    def _kfold_cv(self, X: np.ndarray, y: np.ndarray, 
                 weights: Optional[np.ndarray] = None) -> Dict:
        """K-Fold cross-validation."""
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        
        fold_metrics = []
        for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Fit model
            self.model.fit(X_train, y_train, weights=weights)
            
            # Predict
            y_pred = self.model.predict(X_test)
            
            # Calculate metrics
            metrics = ValidationMetrics(y_test, y_pred, self.model.n_params)
            fold_metrics.append(metrics.calculate_all_metrics())
            
            # Store fold results
            self.cv_results.append({
                'fold': fold,
                'train_idx': train_idx,
                'test_idx': test_idx,
                'y_true': y_test,
                'y_pred': y_pred,
                'metrics': metrics
            })
        
        # Aggregate results
        return self._aggregate_cv_results(fold_metrics)
    
    def _leave_one_out_cv(self, X: np.ndarray, y: np.ndarray,
                         weights: Optional[np.ndarray] = None) -> Dict:
        """Leave-One-Out cross-validation."""
        n_samples = len(X)
        fold_metrics = []
        
        for i in range(n_samples):
            # Leave one out
            train_idx = np.delete(np.arange(n_samples), i)
            test_idx = np.array([i])
            
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Fit model
            self.model.fit(X_train, y_train, weights=weights)
            
            # Predict
            y_pred = self.model.predict(X_test)
            
            # Calculate metrics
            metrics = ValidationMetrics(y_test, y_pred, self.model.n_params)
            fold_metrics.append(metrics.calculate_all_metrics())
            
            self.cv_results.append({
                'fold': i + 1,
                'train_idx': train_idx,
                'test_idx': test_idx,
                'y_true': y_test,
                'y_pred': y_pred,
                'metrics': metrics
            })
        
        return self._aggregate_cv_results(fold_metrics)
    
    def _leave_one_group_out_cv(self, X: np.ndarray, y: np.ndarray,
                               groups: np.ndarray,
                               weights: Optional[np.ndarray] = None) -> Dict:
        """Leave-One-Group-Out cross-validation."""
        unique_groups = np.unique(groups)
        fold_metrics = []
        
        for fold, group in enumerate(unique_groups, 1):
            # Leave one group out
            train_idx = np.where(groups != group)[0]
            test_idx = np.where(groups == group)[0]
            
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Fit model
            self.model.fit(X_train, y_train, weights=weights)
            
            # Predict
            y_pred = self.model.predict(X_test)
            
            # Calculate metrics
            metrics = ValidationMetrics(y_test, y_pred, self.model.n_params)
            fold_metrics.append(metrics.calculate_all_metrics())
            
            self.cv_results.append({
                'fold': fold,
                'group': group,
                'train_idx': train_idx,
                'test_idx': test_idx,
                'y_true': y_test,
                'y_pred': y_pred,
                'metrics': metrics
            })
        
        return self._aggregate_cv_results(fold_metrics)
    
    def _aggregate_cv_results(self, fold_metrics: List[Dict]) -> Dict:
        """Aggregate results across folds."""
        df = pd.DataFrame(fold_metrics)
        
        aggregated = {
            'mean': df.mean().to_dict(),
            'std': df.std().to_dict(),
            'min': df.min().to_dict(),
            'max': df.max().to_dict(),
            'fold_results': df
        }
        
        return aggregated
    
    def plot_cv_results(self, metric: str = 'r2', figsize: Tuple = (10, 6)):
        """Plot cross-validation results for a specific metric."""
        if not self.cv_results:
            raise ValueError("No CV results available. Run cross_validate first.")
        
        fold_values = []
        for result in self.cv_results:
            metrics_dict = result['metrics'].calculate_all_metrics()
            fold_values.append(metrics_dict[metric])
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(range(1, len(fold_values) + 1), fold_values, 'o-', linewidth=2)
        ax.axhline(y=np.mean(fold_values), color='r', linestyle='--', 
                  label=f'Mean: {np.mean(fold_values):.3f}')
        ax.axhline(y=np.mean(fold_values) + np.std(fold_values), 
                  color='r', linestyle=':', alpha=0.5)
        ax.axhline(y=np.mean(fold_values) - np.std(fold_values), 
                  color='r', linestyle=':', alpha=0.5)
        
        ax.set_xlabel('Fold')
        ax.set_ylabel(metric.upper())
        ax.set_title(f'{metric.upper()} across CV Folds')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig, ax

class PredictionValidator:
    """
    Validate model predictions on new or unseen data.
    """
    
    def __init__(self, model, uncertainty_method: str = 'bootstrap'):
        """
        Initialize prediction validator.
        
        Parameters
        ----------
        model : object
            Model to validate
        uncertainty_method : str
            Method for uncertainty estimation: 'bootstrap', 'bayesian', 'analytic'
        """
        self.model = model
        self.uncertainty_method = uncertainty_method
    
    def validate_predictions(self, X_train: np.ndarray, y_train: np.ndarray,
                            X_test: np.ndarray, y_test: np.ndarray,
                            n_bootstrap: int = 1000) -> Dict:
        """
        Validate predictions on test set with uncertainty.
        
        Parameters
        ----------
        X_train, y_train : array-like
            Training data
        X_test, y_test : array-like
            Test data
        n_bootstrap : int
            Number of bootstrap samples
            
        Returns
        -------
        dict
            Validation results with uncertainty
        """
        if self.uncertainty_method == 'bootstrap':
            return self._bootstrap_validation(X_train, y_train, X_test, y_test, n_bootstrap)
        elif self.uncertainty_method == 'analytic':
            return self._analytic_validation(X_train, y_train, X_test, y_test)
        else:
            raise ValueError(f"Unknown uncertainty method: {self.uncertainty_method}")
    
    def _bootstrap_validation(self, X_train: np.ndarray, y_train: np.ndarray,
                             X_test: np.ndarray, y_test: np.ndarray,
                             n_bootstrap: int) -> Dict:
        """Bootstrap-based validation with uncertainty."""
        n_train = len(X_train)
        predictions = []
        
        for i in range(n_bootstrap):
            # Bootstrap sample
            idx = np.random.choice(n_train, n_train, replace=True)
            X_boot = X_train[idx]
            y_boot = y_train[idx]
            
            # Fit on bootstrap sample
            self.model.fit(X_boot, y_boot)
            
            # Predict on test set
            pred = self.model.predict(X_test)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        # Calculate statistics
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        # Confidence intervals
        lower_95 = np.percentile(predictions, 2.5, axis=0)
        upper_95 = np.percentile(predictions, 97.5, axis=0)
        
        # Calculate coverage
        coverage = np.mean((y_test >= lower_95) & (y_test <= upper_95))
        
        # Calculate metrics
        metrics = ValidationMetrics(y_test, mean_pred, self.model.n_params)
        
        return {
            'predictions': predictions,
            'mean_predictions': mean_pred,
            'std_predictions': std_pred,
            'lower_95': lower_95,
            'upper_95': upper_95,
            'coverage_95': coverage,
            'metrics': metrics.calculate_all_metrics(),
            'residuals': y_test - mean_pred
        }
    
    def _analytic_validation(self, X_train: np.ndarray, y_train: np.ndarray,
                            X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Analytic uncertainty propagation."""
        # Fit model on training data
        self.model.fit(X_train, y_train)
        
        # Predict on test set
        y_pred = self.model.predict(X_test)
        
        # Calculate residuals on training set
        y_pred_train = self.model.predict(X_train)
        residuals_train = y_train - y_pred_train
        residual_variance = np.var(residuals_train)
        
        # Simple analytic uncertainty (could be extended based on model)
        n_test = len(X_test)
        uncertainty = np.sqrt(residual_variance * (1 + 1/len(X_train)))
        
        # Confidence intervals
        z_score = 1.96  # 95% confidence
        lower_95 = y_pred - z_score * uncertainty
        upper_95 = y_pred + z_score * uncertainty
        
        # Calculate coverage
        coverage = np.mean((y_test >= lower_95) & (y_test <= upper_95))
        
        # Calculate metrics
        metrics = ValidationMetrics(y_test, y_pred, self.model.n_params)
        
        return {
            'predictions': y_pred,
            'std_predictions': np.full_like(y_pred, uncertainty),
            'lower_95': lower_95,
            'upper_95': upper_95,
            'coverage_95': coverage,
            'metrics': metrics.calculate_all_metrics(),
            'residuals': y_test - y_pred
        }
    
    def prediction_intervals(self, X: np.ndarray, alpha: float = 0.05) -> Dict:
        """
        Calculate prediction intervals for new data.
        
        Parameters
        ----------
        X : array-like
            New data points
        alpha : float
            Significance level (default: 0.05 for 95% CI)
            
        Returns
        -------
        dict
            Prediction intervals
        """
        X = np.asarray(X)
        
        if self.uncertainty_method == 'bootstrap':
            # This would need stored bootstrap samples
            raise NotImplementedError("Bootstrap prediction intervals require stored samples")
        
        elif self.uncertainty_method == 'analytic':
            # Use fitted model parameters
            y_pred = self.model.predict(X)
            
            # Estimate uncertainty (simplified)
            # In practice, this would depend on the model structure
            uncertainty = 0.03 * y_pred  # 3% relative uncertainty
            
            z_score = stats.norm.ppf(1 - alpha/2)
            lower = y_pred - z_score * uncertainty
            upper = y_pred + z_score * uncertainty
            
            return {
                'predictions': y_pred,
                'lower': lower,
                'upper': upper,
                'uncertainty': uncertainty,
                'confidence': 1 - alpha
            }
        
        else:
            raise ValueError(f"Unsupported method: {self.uncertainty_method}")
