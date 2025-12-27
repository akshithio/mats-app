"""
[D]-01 generate_graphs.py:

Phase D: Visualization
Part 01: Comprehensive Graph Generation

This script generates all possible visualizations from the mechanistic analysis
results, creating publication-ready figures with proper styling and labels.

Visualizations:
1. Temporal backward attention comparison (early/late/weighted)
2. Distribution plots for all attention metrics
3. Residual stream ratio comparison
4. Attention entropy comparison
5. Error context attention analysis
6. Correlation heatmaps
7. Statistical summary plots
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import sys
from datetime import datetime

class C:
    HEADER = '\033[95m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

INPUT_FILE = "[B]-01-output.json"
OUTPUT_DIR = "[D]-01-figures/"
LOG_FILE = "[D]-01-logs.txt"

# Plot styling
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
FIGSIZE_SINGLE = (10, 6)
FIGSIZE_DOUBLE = (12, 5)
FIGSIZE_GRID = (15, 10)
DPI = 300

class Logger:
    """Logs output to both console and file, stripping color codes from file."""
    def __init__(self, filename):
        import re
        import os
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8')
        self.re = re
        
    def write(self, message):
        self.terminal.write(message)
        clean_message = self.re.sub(r'\033\[[0-9;]+m', '', message)
        self.log.write(clean_message)
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()
        
    def close(self):
        self.log.close()


class VisualizationGenerator:
    def __init__(self, results_path):
        print(f"{C.OKCYAN}Loading results from {results_path}...{C.ENDC}")
        with open(results_path, 'r') as f:
            self.results = json.load(f)
        
        self.faithful = self.results['faithful']
        self.corrected = self.results['corrected']
        self.metadata = self.results['metadata']
        
        print(f"{C.OKGREEN}Loaded {len(self.faithful)} faithful and {len(self.corrected)} corrected samples{C.ENDC}\n")
        
    def extract_metric(self, data, metric_name):
        """Extract metric values, filtering out None."""
        values = [item[metric_name] for item in data if item.get(metric_name) is not None]
        return np.array(values)
    
    def compute_stats(self, faithful_data, corrected_data):
        """Compute statistical comparison."""
        if len(faithful_data) == 0 or len(corrected_data) == 0:
            return None
        
        t_stat, p_value = stats.ttest_ind(faithful_data, corrected_data)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(faithful_data)-1)*np.std(faithful_data)**2 + 
                              (len(corrected_data)-1)*np.std(corrected_data)**2) / 
                             (len(faithful_data) + len(corrected_data) - 2))
        cohens_d = (np.mean(faithful_data) - np.mean(corrected_data)) / pooled_std if pooled_std > 0 else 0
        
        return {
            'p_value': p_value,
            'cohens_d': cohens_d,
            't_stat': t_stat
        }
    
    def add_significance_stars(self, ax, x1, x2, y, p_value):
        """Add significance stars to plot."""
        if p_value < 0.001:
            sig = '***'
        elif p_value < 0.01:
            sig = '**'
        elif p_value < 0.05:
            sig = '*'
        else:
            sig = 'n.s.'
        
        h = y * 0.05
        ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c='black')
        ax.text((x1+x2)/2, y+h, sig, ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    def plot_temporal_comparison(self):
        """Plot 1: Temporal backward attention comparison."""
        print(f"{C.OKCYAN}Generating temporal attention comparison...{C.ENDC}")
        
        metrics = ['backward_early', 'backward_late', 'backward_weighted']
        labels = ['Early\n(Computational)', 'Late\n(Summary)', 'Weighted\n(Exponential)']
        
        fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
        
        x_pos = np.arange(len(metrics))
        width = 0.35
        
        faithful_means = []
        faithful_sems = []
        corrected_means = []
        corrected_sems = []
        
        for metric in metrics:
            f_data = self.extract_metric(self.faithful, metric)
            c_data = self.extract_metric(self.corrected, metric)
            
            faithful_means.append(np.mean(f_data))
            faithful_sems.append(stats.sem(f_data))
            corrected_means.append(np.mean(c_data))
            corrected_sems.append(stats.sem(c_data))
        
        bars1 = ax.bar(x_pos - width/2, faithful_means, width, yerr=faithful_sems,
                       label='Faithful', alpha=0.8, capsize=5, color='#2E86AB')
        bars2 = ax.bar(x_pos + width/2, corrected_means, width, yerr=corrected_sems,
                       label='Self-Corrected', alpha=0.8, capsize=5, color='#A23B72')
        
        ax.set_xlabel('Temporal Window', fontsize=12, fontweight='bold')
        ax.set_ylabel('Mean Backward Attention', fontsize=12, fontweight='bold')
        ax.set_title('Temporal Backward Attention Analysis\nFaithful vs Self-Corrected', 
                     fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, fontsize=10)
        ax.legend(fontsize=11, loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add significance stars
        for i, metric in enumerate(metrics):
            f_data = self.extract_metric(self.faithful, metric)
            c_data = self.extract_metric(self.corrected, metric)
            stats_result = self.compute_stats(f_data, c_data)
            if stats_result:
                max_y = max(faithful_means[i] + faithful_sems[i], 
                           corrected_means[i] + corrected_sems[i])
                self.add_significance_stars(ax, i-width/2, i+width/2, max_y*1.1, stats_result['p_value'])
        
        plt.tight_layout()
        filename = f"{OUTPUT_DIR}01_temporal_attention_comparison.png"
        plt.savefig(filename, dpi=DPI, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {filename}")
    
    def plot_distribution_comparison(self, metric, title, xlabel, filename):
        """Generic distribution comparison plot."""
        print(f"{C.OKCYAN}Generating {title}...{C.ENDC}")
        
        f_data = self.extract_metric(self.faithful, metric)
        c_data = self.extract_metric(self.corrected, metric)
        
        if len(f_data) == 0 or len(c_data) == 0:
            print(f"{C.WARNING}  Skipping - insufficient data{C.ENDC}")
            return
        
        stats_result = self.compute_stats(f_data, c_data)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE_DOUBLE)
        
        # Violin plot
        parts = ax1.violinplot([f_data, c_data], positions=[0, 1], showmeans=True, 
                               showextrema=True, widths=0.7)
        for pc in parts['bodies']:
            pc.set_alpha(0.7)
        
        ax1.set_xticks([0, 1])
        ax1.set_xticklabels(['Faithful', 'Self-Corrected'])
        ax1.set_ylabel(xlabel, fontsize=11, fontweight='bold')
        ax1.set_title('Distribution Comparison', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add mean values as text
        ax1.text(0, np.mean(f_data), f'{np.mean(f_data):.4f}', 
                ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax1.text(1, np.mean(c_data), f'{np.mean(c_data):.4f}', 
                ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Add significance
        if stats_result:
            max_y = max(np.max(f_data), np.max(c_data))
            self.add_significance_stars(ax1, 0, 1, max_y*1.05, stats_result['p_value'])
        
        # Histogram overlay
        ax2.hist(f_data, bins=30, alpha=0.6, label='Faithful', color='#2E86AB', density=True)
        ax2.hist(c_data, bins=30, alpha=0.6, label='Self-Corrected', color='#A23B72', density=True)
        ax2.set_xlabel(xlabel, fontsize=11, fontweight='bold')
        ax2.set_ylabel('Density', fontsize=11, fontweight='bold')
        ax2.set_title('Histogram Overlay', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # Add stats text
        if stats_result:
            stats_text = (f"t = {stats_result['t_stat']:.3f}\n"
                         f"p = {stats_result['p_value']:.4f}\n"
                         f"d = {stats_result['cohens_d']:.3f}")
            ax2.text(0.95, 0.95, stats_text, transform=ax2.transAxes,
                    fontsize=9, verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}{filename}", dpi=DPI, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {OUTPUT_DIR}{filename}")
    
    def plot_all_metrics_summary(self):
        """Plot 2: Summary bar chart of all metrics."""
        print(f"{C.OKCYAN}Generating all metrics summary...{C.ENDC}")
        
        metrics = [
            ('backward_overall', 'Backward\nOverall'),
            ('backward_early', 'Backward\nEarly'),
            ('backward_late', 'Backward\nLate'),
            ('first_token_attention', 'First\nToken'),
            ('last_token_attention', 'Last\nToken'),
            ('error_context_attention', 'Error\nContext'),
        ]
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        x_pos = np.arange(len(metrics))
        width = 0.35
        
        faithful_means = []
        faithful_sems = []
        corrected_means = []
        corrected_sems = []
        p_values = []
        
        for metric, _ in metrics:
            f_data = self.extract_metric(self.faithful, metric)
            c_data = self.extract_metric(self.corrected, metric)
            
            if len(f_data) > 0 and len(c_data) > 0:
                faithful_means.append(np.mean(f_data))
                faithful_sems.append(stats.sem(f_data))
                corrected_means.append(np.mean(c_data))
                corrected_sems.append(stats.sem(c_data))
                
                stats_result = self.compute_stats(f_data, c_data)
                p_values.append(stats_result['p_value'] if stats_result else 1.0)
            else:
                faithful_means.append(0)
                faithful_sems.append(0)
                corrected_means.append(0)
                corrected_sems.append(0)
                p_values.append(1.0)
        
        bars1 = ax.bar(x_pos - width/2, faithful_means, width, yerr=faithful_sems,
                       label='Faithful', alpha=0.8, capsize=5, color='#2E86AB')
        bars2 = ax.bar(x_pos + width/2, corrected_means, width, yerr=corrected_sems,
                       label='Self-Corrected', alpha=0.8, capsize=5, color='#A23B72')
        
        ax.set_xlabel('Attention Metric', fontsize=12, fontweight='bold')
        ax.set_ylabel('Mean Attention Score', fontsize=12, fontweight='bold')
        ax.set_title('Comprehensive Attention Metrics Comparison', fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([label for _, label in metrics], fontsize=10)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add significance stars
        for i, p_val in enumerate(p_values):
            max_y = max(faithful_means[i] + faithful_sems[i], 
                       corrected_means[i] + corrected_sems[i])
            self.add_significance_stars(ax, i-width/2, i+width/2, max_y*1.1, p_val)
        
        plt.tight_layout()
        filename = f"{OUTPUT_DIR}02_all_metrics_summary.png"
        plt.savefig(filename, dpi=DPI, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {filename}")
    
    def plot_residual_stream_comparison(self):
        """Plot 3: Residual stream ratio comparison."""
        self.plot_distribution_comparison(
            'residual_stream_ratio',
            'Residual Stream Activation Ratio',
            'Residual Stream Ratio (Error/Context)',
            '03_residual_stream_comparison.png'
        )
    
    def plot_attention_entropy_comparison(self):
        """Plot 4: Attention entropy comparison."""
        self.plot_distribution_comparison(
            'error_attention_entropy',
            'Error Position Attention Entropy',
            'Attention Entropy (nats)',
            '04_attention_entropy_comparison.png'
        )
    
    def plot_correlation_heatmap(self):
        """Plot 5: Correlation heatmap between metrics."""
        print(f"{C.OKCYAN}Generating correlation heatmap...{C.ENDC}")
        
        metrics = [
            'backward_overall', 'backward_early', 'backward_late',
            'first_token_attention', 'last_token_attention',
            'error_context_attention', 'residual_stream_ratio',
            'error_attention_entropy'
        ]
        
        labels = [
            'Back Overall', 'Back Early', 'Back Late',
            'First Token', 'Last Token', 'Error Context',
            'Residual', 'Entropy'
        ]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Faithful correlations
        f_data = []
        for metric in metrics:
            values = self.extract_metric(self.faithful, metric)
            f_data.append(values)
        
        # Pad to same length
        max_len = max(len(d) for d in f_data)
        f_data_padded = [np.pad(d, (0, max_len - len(d)), constant_values=np.nan) for d in f_data]
        f_matrix = np.array(f_data_padded)
        f_corr = np.corrcoef(f_matrix)
        
        sns.heatmap(f_corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                   xticklabels=labels, yticklabels=labels, ax=ax1,
                   vmin=-1, vmax=1, cbar_kws={'label': 'Correlation'})
        ax1.set_title('Faithful Samples', fontsize=12, fontweight='bold')
        
        # Corrected correlations
        c_data = []
        for metric in metrics:
            values = self.extract_metric(self.corrected, metric)
            c_data.append(values)
        
        max_len = max(len(d) for d in c_data)
        c_data_padded = [np.pad(d, (0, max_len - len(d)), constant_values=np.nan) for d in c_data]
        c_matrix = np.array(c_data_padded)
        c_corr = np.corrcoef(c_matrix)
        
        sns.heatmap(c_corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                   xticklabels=labels, yticklabels=labels, ax=ax2,
                   vmin=-1, vmax=1, cbar_kws={'label': 'Correlation'})
        ax2.set_title('Self-Corrected Samples', fontsize=12, fontweight='bold')
        
        fig.suptitle('Metric Correlation Heatmaps', fontsize=14, fontweight='bold', y=1.00)
        plt.tight_layout()
        filename = f"{OUTPUT_DIR}05_correlation_heatmaps.png"
        plt.savefig(filename, dpi=DPI, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {filename}")
    
    def plot_effect_sizes(self):
        """Plot 6: Effect sizes (Cohen's d) for all metrics."""
        print(f"{C.OKCYAN}Generating effect sizes plot...{C.ENDC}")
        
        metrics = [
            ('backward_overall', 'Backward Overall'),
            ('backward_early', 'Backward Early'),
            ('backward_late', 'Backward Late'),
            ('backward_weighted', 'Backward Weighted'),
            ('first_token_attention', 'First Token'),
            ('last_token_attention', 'Last Token'),
            ('error_context_attention', 'Error Context'),
            ('residual_stream_ratio', 'Residual Stream'),
            ('error_attention_entropy', 'Entropy')
        ]
        
        effect_sizes = []
        p_values = []
        labels = []
        
        for metric, label in metrics:
            f_data = self.extract_metric(self.faithful, metric)
            c_data = self.extract_metric(self.corrected, metric)
            
            if len(f_data) > 0 and len(c_data) > 0:
                stats_result = self.compute_stats(f_data, c_data)
                if stats_result:
                    effect_sizes.append(stats_result['cohens_d'])
                    p_values.append(stats_result['p_value'])
                    labels.append(label)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = ['#d73027' if abs(d) > 0.8 else '#fee090' if abs(d) > 0.5 else '#91bfdb' 
                  for d in effect_sizes]
        
        y_pos = np.arange(len(labels))
        bars = ax.barh(y_pos, effect_sizes, color=colors, alpha=0.8, edgecolor='black')
        
        # Add significance markers
        for i, (d, p) in enumerate(zip(effect_sizes, p_values)):
            if p < 0.001:
                marker = '***'
            elif p < 0.01:
                marker = '**'
            elif p < 0.05:
                marker = '*'
            else:
                marker = ''
            
            x_pos = d + (0.1 if d > 0 else -0.1)
            ax.text(x_pos, i, f'{d:.2f}{marker}', va='center', fontsize=9, fontweight='bold')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=10)
        ax.set_xlabel("Cohen's d (Effect Size)", fontsize=12, fontweight='bold')
        ax.set_title("Effect Sizes for All Attention Metrics\n(Faithful - Self-Corrected)", 
                     fontsize=14, fontweight='bold', pad=20)
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
        ax.axvline(x=0.5, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
        ax.axvline(x=-0.5, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
        ax.axvline(x=0.8, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
        ax.axvline(x=-0.8, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add legend for effect size interpretation
        legend_text = "Effect Size: |d| < 0.5 (small), 0.5-0.8 (medium), > 0.8 (large)"
        ax.text(0.02, 0.98, legend_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        filename = f"{OUTPUT_DIR}06_effect_sizes.png"
        plt.savefig(filename, dpi=DPI, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {filename}")
    
    def plot_scatter_matrix(self):
        """Plot 7: Scatter matrix of key metrics."""
        print(f"{C.OKCYAN}Generating scatter matrix...{C.ENDC}")
        
        key_metrics = [
            'backward_early',
            'backward_late',
            'error_context_attention',
            'error_attention_entropy'
        ]
        
        labels = ['Early Backward', 'Late Backward', 'Error Context', 'Entropy']
        
        fig, axes = plt.subplots(4, 4, figsize=FIGSIZE_GRID)
        
        for i, metric_i in enumerate(key_metrics):
            for j, metric_j in enumerate(key_metrics):
                ax = axes[i, j]
                
                if i == j:
                    # Diagonal: histograms
                    f_data = self.extract_metric(self.faithful, metric_i)
                    c_data = self.extract_metric(self.corrected, metric_i)
                    
                    ax.hist(f_data, bins=20, alpha=0.6, label='Faithful', color='#2E86AB')
                    ax.hist(c_data, bins=20, alpha=0.6, label='Self-Corrected', color='#A23B72')
                    
                    if i == 0 and j == 0:
                        ax.legend(fontsize=8)
                    
                    ax.set_ylabel('Count', fontsize=8)
                else:
                    # Off-diagonal: scatter plots
                    f_data_x = self.extract_metric(self.faithful, metric_j)
                    f_data_y = self.extract_metric(self.faithful, metric_i)
                    c_data_x = self.extract_metric(self.corrected, metric_j)
                    c_data_y = self.extract_metric(self.corrected, metric_i)
                    
                    # Match lengths
                    min_len_f = min(len(f_data_x), len(f_data_y))
                    min_len_c = min(len(c_data_x), len(c_data_y))
                    
                    ax.scatter(f_data_x[:min_len_f], f_data_y[:min_len_f], 
                              alpha=0.5, s=20, label='Faithful', color='#2E86AB')
                    ax.scatter(c_data_x[:min_len_c], c_data_y[:min_len_c], 
                              alpha=0.5, s=20, label='Self-Corrected', color='#A23B72')
                
                if i == len(key_metrics) - 1:
                    ax.set_xlabel(labels[j], fontsize=9)
                else:
                    ax.set_xticklabels([])
                
                if j == 0:
                    ax.set_ylabel(labels[i], fontsize=9)
                else:
                    ax.set_yticklabels([])
                
                ax.tick_params(labelsize=7)
                ax.grid(True, alpha=0.2)
        
        fig.suptitle('Scatter Matrix: Key Attention Metrics', fontsize=14, fontweight='bold')
        plt.tight_layout()
        filename = f"{OUTPUT_DIR}07_scatter_matrix.png"
        plt.savefig(filename, dpi=DPI, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {filename}")
    
    def plot_ratio_comparison(self):
        """Plot 8: Ratio comparison (Self-Corrected / Faithful)."""
        print(f"{C.OKCYAN}Generating ratio comparison...{C.ENDC}")
        
        metrics = [
            ('backward_overall', 'Overall'),
            ('backward_early', 'Early'),
            ('backward_late', 'Late'),
            ('first_token_attention', 'First Token'),
            ('last_token_attention', 'Last Token'),
            ('error_context_attention', 'Error Context')
        ]
        
        ratios = []
        labels = []
        
        for metric, label in metrics:
            f_data = self.extract_metric(self.faithful, metric)
            c_data = self.extract_metric(self.corrected, metric)
            
            if len(f_data) > 0 and len(c_data) > 0:
                ratio = np.mean(c_data) / np.mean(f_data)
                ratios.append(ratio)
                labels.append(label)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = ['#A23B72' if r > 1 else '#2E86AB' for r in ratios]
        y_pos = np.arange(len(labels))
        bars = ax.barh(y_pos, ratios, color=colors, alpha=0.8, edgecolor='black')
        
        # Add ratio values
        for i, r in enumerate(ratios):
            x_pos = r + (0.05 if r > 1 else -0.05)
            ha = 'left' if r > 1 else 'right'
            ax.text(x_pos, i, f'{r:.2f}x', va='center', ha=ha, fontsize=10, fontweight='bold')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=11)
        ax.set_xlabel('Ratio (Self-Corrected / Faithful)', fontsize=12, fontweight='bold')
        ax.set_title('Attention Metric Ratios\nSelf-Corrected vs Faithful', 
                     fontsize=14, fontweight='bold', pad=20)
        ax.axvline(x=1, color='black', linestyle='--', linewidth=2, label='Equal')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        filename = f"{OUTPUT_DIR}08_ratio_comparison.png"
        plt.savefig(filename, dpi=DPI, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {filename}")
    
    def generate_all_plots(self):
        """Generate all visualizations."""
        print(f"{C.HEADER}{'='*80}{C.ENDC}")
        print(f"{C.HEADER}Generating All Visualizations{C.ENDC}")
        print(f"{C.HEADER}{'='*80}{C.ENDC}\n")
        
        self.plot_temporal_comparison()
        self.plot_all_metrics_summary()
        self.plot_residual_stream_comparison()
        self.plot_attention_entropy_comparison()
        self.plot_correlation_heatmap()
        self.plot_effect_sizes()
        self.plot_scatter_matrix()
        self.plot_ratio_comparison()
        
        print(f"\n{C.OKGREEN}All visualizations generated successfully!{C.ENDC}")
        print(f"{C.BOLD}Output directory: {OUTPUT_DIR}{C.ENDC}")


def main():
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    logger = Logger(LOG_FILE)
    sys.stdout = logger
    
    try:
        print(f"Visualization Generation Log")
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}\n")
        
        generator = VisualizationGenerator(INPUT_FILE)
        generator.generate_all_plots()
        
        print(f"\n{C.HEADER}{'='*80}{C.ENDC}")
        print(f"{C.BOLD}Summary:{C.ENDC}")
        print(f"  • 8 visualizations generated")
        print(f"  • Output directory: {OUTPUT_DIR}")
        print(f"  • Log file: {LOG_FILE}")
        print(f"{C.HEADER}{'='*80}{C.ENDC}")
        
    finally:
        sys.stdout = logger.terminal
        logger.close()
        print(f"\n{C.OKGREEN}Full output saved to {LOG_FILE}{C.ENDC}")


if __name__ == "__main__":
    main()