"""
Visualization tools for holographic chemistry.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class PeriodicTablePlotter:
    """
    Plot periodic table with holographic predictions.
    """
    
    def __init__(self, element_data: pd.DataFrame):
        """
        Initialize plotter with element data.
        
        Parameters
        ----------
        element_data : DataFrame
            DataFrame with columns: Z, symbol, period, group, radius
        """
        self.element_data = element_data
        self.colors = {
            'alkali': '#FF6666',
            'alkaline': '#FFDEAD',
            'transition': '#FFB347',
            'basic': '#98FB98',
            'nonmetal': '#ADD8E6',
            'halogen': '#FFB6C1',
            'noble': '#DDA0DD',
            'lanthanide': '#C9A0DC',
            'actinide': '#C9A0DC'
        }
    
    def plot_periodic_table(self, values: Optional[np.ndarray] = None,
                           value_name: str = "Radius (Å)",
                           cmap: str = 'viridis',
                           figsize: Tuple = (16, 9)) -> Tuple[Figure, Axes]:
        """
        Plot periodic table with values.
        
        Parameters
        ----------
        values : array-like, optional
            Values to plot (e.g., radii or residuals)
        value_name : str
            Name for colorbar
        cmap : str
            Colormap name
        figsize : tuple
            Figure size
            
        Returns
        -------
        fig, ax
            Matplotlib figure and axes
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Periodic table layout
        periods = self.element_data['period'].max()
        groups = self.element_data['group'].max()
        
        # Create grid
        x_positions = {}
        y_positions = {}
        
        for _, row in self.element_data.iterrows():
            period = row['period']
            group = row['group']
            
            # Adjust for lanthanides/actinides
            if period > 6 and group > 2:
                x = group + 14  # Shift right for f-block
                y = period + 2.5  # Shift down
            else:
                x = group
                y = period
            
            x_positions[row['symbol']] = x
            y_positions[row['symbol']] = y
        
        # Plot elements
        if values is not None:
            norm = plt.Normalize(values.min(), values.max())
            cmap = plt.cm.get_cmap(cmap)
        else:
            # Color by block
            for _, row in self.element_data.iterrows():
                symbol = row['symbol']
                block = self._get_block(row['Z'])
                color = self._get_block_color(block)
                
                x = x_positions[symbol]
                y = y_positions[symbol]
                
                ax.add_patch(plt.Rectangle((x-0.5, y-0.5), 1, 1,
                                         facecolor=color, edgecolor='black'))
                
                # Add symbol
                ax.text(x, y, symbol, ha='center', va='center',
                       fontweight='bold', fontsize=10)
        
        ax.set_xlim(0, groups + 16)
        ax.set_ylim(0, periods + 4)
        ax.set_aspect('equal')
        ax.invert_yaxis()  # Period 1 at top
        ax.axis('off')
        
        # Add colorbar if values provided
        if values is not None:
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
            cbar.set_label(value_name, fontsize=12)
        
        # Add title
        ax.set_title('Periodic Table', fontsize=16, fontweight='bold')
        
        return fig, ax
    
    def plot_trends(self, Z: np.ndarray, observed: np.ndarray,
                   predicted: np.ndarray, figsize: Tuple = (12, 8)) -> Figure:
        """
        Plot observed vs predicted trends.
        
        Parameters
        ----------
        Z : array-like
            Atomic numbers
        observed : array-like
            Observed radii
        predicted : array-like
            Predicted radii
            
        Returns
        -------
        Figure
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # 1. Observed vs Predicted
        ax = axes[0, 0]
        ax.scatter(observed, predicted, alpha=0.7, edgecolors='k', linewidth=0.5)
        
        # Perfect prediction line
        min_val = min(observed.min(), predicted.min())
        max_val = max(observed.max(), predicted.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)
        
        ax.set_xlabel('Observed Radius (Å)')
        ax.set_ylabel('Predicted Radius (Å)')
        ax.set_title('Observed vs Predicted')
        ax.grid(True, alpha=0.3)
        
        # 2. Residuals vs Atomic Number
        ax = axes[0, 1]
        residuals = observed - predicted
        ax.scatter(Z, residuals, alpha=0.7, edgecolors='k', linewidth=0.5)
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        
        # Highlight noble gases
        noble_mask = np.isin(Z, [2, 10, 18, 36, 54, 86])
        if np.any(noble_mask):
            ax.scatter(Z[noble_mask], residuals[noble_mask],
                      color='purple', alpha=0.9, label='Noble Gases',
                      edgecolors='k', linewidth=0.5)
            ax.legend()
        
        ax.set_xlabel('Atomic Number (Z)')
        ax.set_ylabel('Residual (Å)')
        ax.set_title('Residuals vs Atomic Number')
        ax.grid(True, alpha=0.3)
        
        # 3. Period trends
        ax = axes[1, 0]
        periods = self.element_data.set_index('Z')['period'].loc[Z].values
        
        for period in np.unique(periods):
            mask = periods == period
            if np.sum(mask) > 1:
                ax.plot(Z[mask], observed[mask], 'o-', label=f'Period {period}')
        
        ax.set_xlabel('Atomic Number (Z)')
        ax.set_ylabel('Radius (Å)')
        ax.set_title('Period Trends')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Group trends (log scale)
        ax = axes[1, 1]
        groups = self.element_data.set_index('Z')['group'].loc[Z].values
        
        for group in np.unique(groups):
            if group in [1, 2, 17, 18]:  # Main groups
                mask = groups == group
                if np.sum(mask) > 1:
                    ax.plot(Z[mask], observed[mask], 's-', label=f'Group {group}')
        
        ax.set_xlabel('Atomic Number (Z)')
        ax.set_ylabel('Radius (Å)')
        ax.set_title('Group Trends (Selected Groups)')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_contributions(self, Z: np.ndarray, contributions: Dict,
                          figsize: Tuple = (14, 8)) -> Figure:
        """
        Plot contribution analysis.
        
        Parameters
        ----------
        Z : array-like
            Atomic numbers
        contributions : dict
            Dictionary with contribution arrays
            
        Returns
        -------
        Figure
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Sort by Z
        sort_idx = np.argsort(Z)
        Z_sorted = Z[sort_idx]
        
        # 1. Absolute contributions
        ax = axes[0, 0]
        width = 0.8
        bottom = np.zeros_like(Z_sorted)
        
        for name, array in [('Mean Field', contributions['mean_field']),
                           ('QED/Entanglement', contributions['qed_correction']),
                           ('Noble Correction', contributions['noble_correction'])]:
            values = array[sort_idx]
            ax.bar(Z_sorted, values, width, bottom=bottom, label=name)
            bottom += values
        
        ax.set_xlabel('Atomic Number (Z)')
        ax.set_ylabel('Contribution (Å)')
        ax.set_title('Absolute Contributions')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Percentage contributions
        ax = axes[0, 1]
        bottom = np.zeros_like(Z_sorted)
        
        for name, array in [('Mean Field', contributions['mean_field_pct']),
                           ('QED/Entanglement', contributions['qed_correction_pct']),
                           ('Noble Correction', contributions['noble_correction_pct'])]:
            values = array[sort_idx]
            ax.bar(Z_sorted, values, width, bottom=bottom, label=name)
            bottom += values
        
        ax.set_xlabel('Atomic Number (Z)')
        ax.set_ylabel('Contribution (%)')
        ax.set_title('Percentage Contributions')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. QED contribution vs Z
        ax = axes[1, 0]
        ax.scatter(Z, contributions['qed_correction_pct'], alpha=0.7,
                  edgecolors='k', linewidth=0.5)
        
        # Fit trend
        mask = Z > 10  # Exclude very light elements
        if np.sum(mask) > 2:
            z_fit = np.linspace(10, Z.max(), 100)
            # Simple polynomial fit
            coeffs = np.polyfit(Z[mask], contributions['qed_correction_pct'][mask], 2)
            p = np.poly1d(coeffs)
            ax.plot(z_fit, p(z_fit), 'r--', alpha=0.7, label='Trend')
            ax.legend()
        
        ax.set_xlabel('Atomic Number (Z)')
        ax.set_ylabel('QED/Entanglement Contribution (%)')
        ax.set_title('QED Contribution Trend')
        ax.grid(True, alpha=0.3)
        
        # 4. Noble gas analysis
        ax = axes[1, 1]
        noble_mask = np.isin(Z, [2, 10, 18, 36, 54, 86])
        
        if np.any(noble_mask):
            noble_Z = Z[noble_mask]
            noble_corr = contributions['noble_correction'][noble_mask]
            noble_total = contributions['total'][noble_mask]
            
            ax.bar(noble_Z, noble_corr / noble_total * 100,
                  color='purple', alpha=0.7, edgecolor='black')
            
            # Add element labels
            symbols = self.element_data.set_index('Z')['symbol'].loc[noble_Z].values
            for i, (z, sym) in enumerate(zip(noble_Z, symbols)):
                ax.text(z, (noble_corr[i] / noble_total[i] * 100) + 0.5,
                       sym, ha='center', fontweight='bold')
        
        ax.set_xlabel('Atomic Number (Z)')
        ax.set_ylabel('Noble Correction (%)')
        ax.set_title('Noble Gas Correction Analysis')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def _get_block(self, Z: int) -> str:
        """Get block for element."""
        if Z == 1 or Z == 2:
            return 'nonmetal'
        elif 3 <= Z <= 10:
            return 'basic'
        elif 11 <= Z <= 18:
            return 'basic'
        elif 19 <= Z <= 20:
            return 'alkaline'
        elif 21 <= Z <= 30:
            return 'transition'
        elif 31 <= Z <= 36:
            return 'basic'
        elif 37 <= Z <= 38:
            return 'alkaline'
        elif 39 <= Z <= 48:
            return 'transition'
        elif 49 <= Z <= 54:
            return 'basic'
        elif 55 <= Z <= 56:
            return 'alkali'
        elif 57 <= Z <= 71:
            return 'lanthanide'
        elif 72 <= Z <= 80:
            return 'transition'
        elif 81 <= Z <= 86:
            return 'basic'
        elif 87 <= Z <= 88:
            return 'alkali'
        elif 89 <= Z <= 103:
            return 'actinide'
        else:
            return 'basic'
    
    def _get_block_color(self, block: str) -> str:
        """Get color for block."""
        return self.colors.get(block, '#FFFFFF')

class HolographicMapVisualizer:
    """
    Visualize holographic mapping concepts.
    """
    
    def __init__(self):
        """Initialize visualizer."""
        self.figure_templates = {}
    
    def create_holographic_diagram(self, figsize: Tuple = (12, 8)) -> Figure:
        """
        Create diagram illustrating holographic mapping.
        
        Returns
        -------
        Figure
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Left: Boundary theory (CFT)
        ax = axes[0]
        self._plot_boundary_theory(ax)
        
        # Right: Bulk theory (AdS gravity)
        ax = axes[1]
        self._plot_bulk_theory(ax)
        
        # Add connecting arrows
        fig.text(0.45, 0.5, '⟷', fontsize=40, ha='center', va='center')
        fig.text(0.45, 0.55, 'Holographic\nDuality', fontsize=12,
                ha='center', va='bottom', fontweight='bold')
        
        plt.suptitle('Holographic Correspondence in Chemistry', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def _plot_boundary_theory(self, ax: Axes):
        """Plot boundary theory representation."""
        # Create lattice points
        x = np.linspace(0, 10, 11)
        y = np.linspace(0, 10, 11)
        X, Y = np.meshgrid(x, y)
        
        # Plot lattice
        ax.scatter(X.flatten(), Y.flatten(), color='blue', s=50, alpha=0.7)
        
        # Add some connections
        for i in range(10):
            ax.plot([i, i+1], [5, 5], 'k-', alpha=0.3)
            ax.plot([5, 5], [i, i+1], 'k-', alpha=0.3)
        
        # Add operator
        ax.scatter(5, 5, color='red', s=200, marker='*', edgecolors='black', linewidth=2)
        ax.text(5, 5.5, r'$\mathcal{O}(x)$', fontsize=14, ha='center', fontweight='bold')
        
        ax.set_xlim(-1, 11)
        ax.set_ylim(-1, 11)
        ax.set_aspect('equal')
        ax.set_title('Boundary Theory\n(Quantum Chemistry)', fontsize=12, fontweight='bold')
        ax.axis('off')
    
    def _plot_bulk_theory(self, ax: Axes):
        """Plot bulk theory representation."""
        # Create curved spacetime
        z = np.linspace(0.1, 5, 50)
        r = np.linspace(0, 10, 50)
        R, Z = np.meshgrid(r, z)
        
        # Warped metric
        X = R * np.cos(Z/2)
        Y = R * np.sin(Z/2)
        
        # Plot surface
        ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6)
        
        # Add radial direction
        r_line = np.linspace(0, 8, 100)
        z_line = np.ones_like(r_line) * 0.5
        ax.plot(r_line * np.cos(0.25), r_line * np.sin(0.25), z_line,
               'r-', linewidth=3, label='Radial Direction')
        
        # Add boundary
        ax.plot(r * np.cos(2.5), r * np.sin(2.5), np.ones_like(r) * 5,
               'k--', linewidth=2, label='Boundary (Z=0)')
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('Radial Coordinate')
        ax.set_title('Bulk Theory\n(Higher-Dimensional Geometry)', fontsize=12, fontweight='bold')
        ax.legend()
    
    def plot_entanglement_entropy(self, Z: np.ndarray, S_ent: np.ndarray,
                                 r_c: np.ndarray, figsize: Tuple = (10, 8)) -> Figure:
        """
        Plot entanglement entropy relationship.
        
        Parameters
        ----------
        Z : array-like
            Atomic numbers
        S_ent : array-like
            Entanglement entropy proxies
        r_c : array-like
            Covalent radii
            
        Returns
        -------
        Figure
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # 1. S_ent vs Z
        ax = axes[0, 0]
        ax.scatter(Z, S_ent, alpha=0.7, edgecolors='k', linewidth=0.5)
        
        # Highlight noble gases
        noble_mask = np.isin(Z, [2, 10, 18, 36, 54, 86])
        if np.any(noble_mask):
            ax.scatter(Z[noble_mask], S_ent[noble_mask],
                      color='purple', alpha=0.9, label='Noble Gases',
                      edgecolors='k', linewidth=0.5, s=80)
            ax.legend()
        
        ax.set_xlabel('Atomic Number (Z)')
        ax.set_ylabel('Entanglement Entropy Proxy')
        ax.set_title('Entanglement vs Atomic Number')
        ax.grid(True, alpha=0.3)
        
        # 2. S_ent vs r_c
        ax = axes[0, 1]
        ax.scatter(S_ent, r_c, alpha=0.7, edgecolors='k', linewidth=0.5)
        
        # Linear fit
        if len(S_ent) > 2:
            coeffs = np.polyfit(S_ent, r_c, 1)
            p = np.poly1d(coeffs)
            S_fit = np.linspace(S_ent.min(), S_ent.max(), 100)
            ax.plot(S_fit, p(S_fit), 'r--', alpha=0.7,
                   label=f'R = {np.corrcoef(S_ent, r_c)[0,1]:.3f}')
            ax.legend()
        
        ax.set_xlabel('Entanglement Entropy Proxy')
        ax.set_ylabel('Covalent Radius (Å)')
        ax.set_title('Entanglement vs Radius')
        ax.grid(True, alpha=0.3)
        
        # 3. Holographic relation test
        ax = axes[1, 0]
        # Calculate G_N(Z) = 1/(1 + κ α² Z^(2/3))
        alpha = 1/137.036
        kappa = 9.8
        G_N = 1 / (1 + kappa * alpha ** 2 * np.power(Z, 2/3))
        
        lhs = r_c
        rhs = S_ent / G_N
        
        ax.scatter(rhs, lhs, alpha=0.7, edgecolors='k', linewidth=0.5)
        
        # Perfect line
        min_val = min(rhs.min(), lhs.min())
        max_val = max(rhs.max(), lhs.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)
        
        correlation = np.corrcoef(rhs, lhs)[0, 1]
        ax.text(0.05, 0.95, f'Correlation: {correlation:.3f}',
               transform=ax.transAxes, fontsize=12,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_xlabel(r'$S_{\mathrm{ent}} / G_N(Z)$')
        ax.set_ylabel(r'$r_c$ (Å)')
        ax.set_title('Holographic Relation Test')
        ax.grid(True, alpha=0.3)
        
        # 4. Log-log plot for scaling
        ax = axes[1, 1]
        ax.loglog(Z, S_ent, 'o-', alpha=0.7, linewidth=2)
        
        # Power law fit
        if len(Z) > 2:
            log_Z = np.log(Z[Z > 0])
            log_S = np.log(S_ent[Z > 0])
            coeffs = np.polyfit(log_Z, log_S, 1)
            power = coeffs[0]
            
            Z_fit = np.logspace(np.log10(Z.min()), np.log10(Z.max()), 100)
            S_fit = np.exp(coeffs[1]) * Z_fit ** power
            
            ax.loglog(Z_fit, S_fit, 'r--', alpha=0.7,
                     label=f'Power law: ~Z^{power:.2f}')
            ax.legend()
        
        ax.set_xlabel('Atomic Number (Z)')
        ax.set_ylabel('Entanglement Entropy Proxy')
        ax.set_title('Scaling Behavior (Log-Log)')
        ax.grid(True, alpha=0.3, which='both')
        
        plt.suptitle('Entanglement Entropy Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return fig

class ContributionAnalyzer:
    """
    Analyze and visualize model contributions.
    """
    
    def __init__(self, model):
        """
        Initialize analyzer.
        
        Parameters
        ----------
        model : HolographicModel
            Model to analyze
        """
        self.model = model
    
    def create_interactive_plot(self, Z_range: Tuple = (1, 86)) -> go.Figure:
        """
        Create interactive plot of contributions.
        
        Parameters
        ----------
        Z_range : tuple
            Range of atomic numbers
            
        Returns
        -------
        go.Figure
            Plotly figure
        """
        Z = np.arange(Z_range[0], Z_range[1] + 1)
        
        # Calculate contributions
        contributions = self.model.contribution_analysis(Z)
        
        # Create figure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Absolute Contributions',
                          'Percentage Contributions',
                          'QED Contribution Trend',
                          'Noble Gas Analysis'),
            specs=[[{'type': 'xy'}, {'type': 'xy'}],
                  [{'type': 'xy'}, {'type': 'xy'}]]
        )
        
        # 1. Absolute contributions (stacked bar)
        fig.add_trace(
            go.Bar(x=Z, y=contributions['mean_field'],
                  name='Mean Field', marker_color='blue'),
            row=1, col=1
        )
        fig.add_trace(
            go.Bar(x=Z, y=contributions['qed_correction'],
                  name='QED/Entanglement', marker_color='green'),
            row=1, col=1
        )
        fig.add_trace(
            go.Bar(x=Z, y=contributions['noble_correction'],
                  name='Noble Correction', marker_color='purple'),
            row=1, col=1
        )
        
        fig.update_xaxes(title_text='Atomic Number (Z)', row=1, col=1)
        fig.update_yaxes(title_text='Contribution (Å)', row=1, col=1)
        
        # 2. Percentage contributions (stacked bar)
        fig.add_trace(
            go.Bar(x=Z, y=contributions['mean_field_pct'],
                  name='Mean Field %', marker_color='blue'),
            row=1, col=2
        )
        fig.add_trace(
            go.Bar(x=Z, y=contributions['qed_correction_pct'],
                  name='QED/Entanglement %', marker_color='green'),
            row=1, col=2
        )
        fig.add_trace(
            go.Bar(x=Z, y=contributions['noble_correction_pct'],
                  name='Noble Correction %', marker_color='purple'),
            row=1, col=2
        )
        
        fig.update_xaxes(title_text='Atomic Number (Z)', row=1, col=2)
        fig.update_yaxes(title_text='Contribution (%)', row=1, col=2)
        
        # 3. QED contribution trend
        fig.add_trace(
            go.Scatter(x=Z, y=contributions['qed_correction_pct'],
                      mode='markers', name='Data',
                      marker=dict(color='green', size=8)),
            row=2, col=1
        )
        
        # Add trend line
        if len(Z) > 10:
            mask = Z > 10
            coeffs = np.polyfit(Z[mask], contributions['qed_correction_pct'][mask], 2)
            p = np.poly1d(coeffs)
            Z_fit = np.linspace(10, Z.max(), 100)
            fig.add_trace(
                go.Scatter(x=Z_fit, y=p(Z_fit),
                          mode='lines', name='Trend',
                          line=dict(color='red', dash='dash')),
                row=2, col=1
            )
        
        fig.update_xaxes(title_text='Atomic Number (Z)', row=2, col=1)
        fig.update_yaxes(title_text='QED Contribution (%)', row=2, col=1)
        
        # 4. Noble gas analysis
        noble_mask = self.model.is_noble_gas(Z)
        if np.any(noble_mask):
            noble_Z = Z[noble_mask]
            noble_contrib = contributions['noble_correction'][noble_mask]
            noble_total = contributions['total'][noble_mask]
            
            fig.add_trace(
                go.Bar(x=noble_Z, y=noble_contrib / noble_total * 100,
                      name='Noble Correction',
                      marker_color='purple'),
                row=2, col=2
            )
        
        fig.update_xaxes(title_text='Atomic Number (Z)', row=2, col=2)
        fig.update_yaxes(title_text='Noble Correction (%)', row=2, col=2)
        
        # Update layout
        fig.update_layout(
            title='Holographic Model Contribution Analysis',
            height=800,
            showlegend=True,
            barmode='stack'
        )
        
        return fig
    
    def create_prediction_dashboard(self, Z: np.ndarray, observed: np.ndarray,
                                   predicted: np.ndarray) -> go.Figure:
        """
        Create interactive prediction dashboard.
        
        Parameters
        ----------
        Z : array-like
            Atomic numbers
        observed : array-like
            Observed radii
        predicted : array-like
            Predicted radii
            
        Returns
        -------
        go.Figure
            Plotly figure
        """
        residuals = observed - predicted
        relative_error = np.abs(residuals / observed) * 100
        
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Observed vs Predicted',
                          'Residuals Distribution',
                          'Error vs Atomic Number',
                          'Relative Error Analysis',
                          'Period Trends',
                          'Group Trends'),
            specs=[[{'type': 'xy'}, {'type': 'xy'}],
                  [{'type': 'xy'}, {'type': 'xy'}],
                  [{'type': 'xy'}, {'type': 'xy'}]],
            vertical_spacing=0.1
        )
        
        # 1. Observed vs Predicted
        fig.add_trace(
            go.Scatter(x=observed, y=predicted, mode='markers',
                      name='Elements',
                      marker=dict(size=8, color=Z, colorscale='Viridis',
                                 showscale=True,
                                 colorbar=dict(title='Z'))),
            row=1, col=1
        )
        
        # Perfect line
        min_val = min(observed.min(), predicted.min())
        max_val = max(observed.max(), predicted.max())
        fig.add_trace(
            go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                      mode='lines', name='Perfect',
                      line=dict(color='red', dash='dash')),
            row=1, col=1
        )
        
        fig.update_xaxes(title_text='Observed (Å)', row=1, col=1)
        fig.update_yaxes(title_text='Predicted (Å)', row=1, col=1)
        
        # 2. Residuals distribution
        fig.add_trace(
            go.Histogram(x=residuals, nbinsx=30,
                        name='Residuals',
                        marker_color='blue'),
            row=1, col=2
        )
        
        # Normal distribution overlay
        mu, sigma = residuals.mean(), residuals.std()
        x_norm = np.linspace(residuals.min(), residuals.max(), 100)
        y_norm = 1/(sigma * np.sqrt(2*np.pi)) * np.exp(-0.5*((x_norm-mu)/sigma)**2)
        y_norm = y_norm / y_norm.max() * np.histogram(residuals, bins=30)[0].max()
        
        fig.add_trace(
            go.Scatter(x=x_norm, y=y_norm,
                      mode='lines', name='Normal',
                      line=dict(color='red', width=2)),
            row=1, col=2
        )
        
        fig.update_xaxes(title_text='Residual (Å)', row=1, col=2)
        fig.update_yaxes(title_text='Count', row=1, col=2)
        
        # 3. Error vs Z
        fig.add_trace(
            go.Scatter(x=Z, y=np.abs(residuals), mode='markers',
                      name='Absolute Error',
                      marker=dict(size=8, color=relative_error,
                                 colorscale='RdBu', showscale=True,
                                 colorbar=dict(title='Relative Error %'))),
            row=2, col=1
        )
        
        fig.update_xaxes(title_text='Atomic Number (Z)', row=2, col=1)
        fig.update_yaxes(title_text='Absolute Error (Å)', row=2, col=1)
        
        # 4. Relative error analysis
        fig.add_trace(
            go.Box(y=relative_error, name='All Elements',
                  boxpoints='outliers'),
            row=2, col=2
        )
        
        # Add by group boxes
        # (Would need group information)
        
        fig.update_yaxes(title_text='Relative Error (%)', row=2, col=2)
        
        # 5. Period trends (placeholder)
        fig.add_trace(
            go.Scatter(x=[], y=[], mode='lines'),
            row=3, col=1
        )
        
        fig.update_xaxes(title_text='Atomic Number (Z)', row=3, col=1)
        fig.update_yaxes(title_text='Radius (Å)', row=3, col=1)
        
        # 6. Group trends (placeholder)
        fig.add_trace(
            go.Scatter(x=[], y=[], mode='lines'),
            row=3, col=2
        )
        
        fig.update_xaxes(title_text='Atomic Number (Z)', row=3, col=2)
        fig.update_yaxes(title_text='Radius (Å)', row=3, col=2)
        
        # Update layout
        fig.update_layout(
            title='Prediction Analysis Dashboard',
            height=1200,
            showlegend=True
        )
        
        return fig
