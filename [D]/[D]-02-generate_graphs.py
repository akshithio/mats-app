"""
[D]-02 generate_graphs.py:

Phase D: Visualization
Part 02: Embedding Geometry Analysis Visualizations

This script generates visualizations for the internal belief analysis
via embedding geometry (thought-action decoupling investigation).

Visualizations:
1. Belief bias comparison (sim_error - sim_truth)
2. Detailed similarity breakdown (error vs truth separately)
3. Distribution comparisons (violin + histogram)
4. Individual similarity scatter plots
5. Belief bias by problem (trajectory plots)
6. Statistical summary with effect sizes
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

INPUT_FILE = "[B]-02-output.json"
OUTPUT_DIR = "[D]-02-figures/"
LOG_FILE = "[D]-02-logs.txt"

# Plot styling
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
FIGSIZE_SINGLE = (10, 6)
FIGSIZE_DOUBLE = (12, 5)
FIGSIZE_LARGE = (14, 8)
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


class EmbeddingVisualizationGenerator:
    def __init__(self, results_path):
        print(f"{C.OKCYAN}Loading results from {results_path}...{C.ENDC}")
        with open(results_path, 'r') as f:
            self.results = json.load(f)
        
        # Extract successful analyses only
        self.faithful = [r for r in self.results['faithful'] if r['success']]
        self.corrected = [r for r in self.results['corrected'] if r['success']]
        self.metadata = self.results['metadata']
        
        print(f"{C.OKGREEN}Loaded {len(self.faithful)} faithful and {len(self.corrected)} corrected samples{C.ENDC}\n")
        
    def extract_metric(self, data, metric_name):
        """Extract metric values."""
        values = [item[metric_name] for item in data]
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
    
    def plot_belief_bias_comparison(self):
        """Plot 1: Main belief bias comparison."""
        print(f"{C.OKCYAN}Generating belief bias comparison...{C.ENDC}")
        
        f_bias = self.extract_metric(self.faithful, 'belief_bias')
        c_bias = self.extract_metric(self.corrected, 'belief_bias')
        
        stats_result = self.compute_stats(f_bias, c_bias)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE_DOUBLE)
        
        # Bar plot with error bars
        categories = ['Faithful', 'Self-Corrected']
        means = [np.mean(f_bias), np.mean(c_bias)]
        sems = [stats.sem(f_bias), stats.sem(c_bias)]
        colors = ['#E63946', '#06A77D']
        
        bars = ax1.bar(categories, means, yerr=sems, color=colors, alpha=0.8, 
                      capsize=8, edgecolor='black', linewidth=1.5)
        
        ax1.axhline(y=0, color='black', linestyle='--', linewidth=2, alpha=0.7)
        ax1.set_ylabel('Belief Bias\n(sim_error - sim_truth)', fontsize=12, fontweight='bold')
        ax1.set_title('[B-02] Belief Bias Comparison\nPositive = Aligned with Error, Negative = Aligned with Truth', 
                     fontsize=13, fontweight='bold', pad=15)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for i, (bar, mean, sem) in enumerate(zip(bars, means, sems)):
            height = bar.get_height()
            label = f'{mean:+.4f}\n±{sem:.4f}'
            y_pos = height + sem if height > 0 else height - sem
            va = 'bottom' if height > 0 else 'top'
            ax1.text(bar.get_x() + bar.get_width()/2, y_pos, label,
                    ha='center', va=va, fontsize=10, fontweight='bold')
        
        # Add significance
        if stats_result:
            max_y = max(means[0] + sems[0], means[1] + sems[1])
            self.add_significance_stars(ax1, 0, 1, max_y*1.15, stats_result['p_value'])
        
        # Distribution comparison
        positions = [1, 2]
        parts = ax2.violinplot([f_bias, c_bias], positions=positions, 
                              showmeans=True, showextrema=True, widths=0.7)
        
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.7)
        
        ax2.axhline(y=0, color='black', linestyle='--', linewidth=2, alpha=0.7)
        ax2.set_xticks(positions)
        ax2.set_xticklabels(categories)
        ax2.set_ylabel('Belief Bias', fontsize=12, fontweight='bold')
        ax2.set_title('[B-02] Distribution Comparison', fontsize=13, fontweight='bold', pad=15)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add stats text
        if stats_result:
            stats_text = (f"t = {stats_result['t_stat']:.3f}\n"
                         f"p = {stats_result['p_value']:.4f}\n"
                         f"d = {stats_result['cohens_d']:.3f}")
            ax2.text(0.05, 0.95, stats_text, transform=ax2.transAxes,
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        
        plt.tight_layout()
        filename = f"{OUTPUT_DIR}01_belief_bias_comparison.png"
        plt.savefig(filename, dpi=DPI, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {filename}")
    
    def plot_detailed_similarity_breakdown(self):
        """Plot 2: Similarity to error vs truth separately."""
        print(f"{C.OKCYAN}Generating detailed similarity breakdown...{C.ENDC}")
        
        f_sim_error = self.extract_metric(self.faithful, 'sim_to_error')
        f_sim_truth = self.extract_metric(self.faithful, 'sim_to_truth')
        c_sim_error = self.extract_metric(self.corrected, 'sim_to_error')
        c_sim_truth = self.extract_metric(self.corrected, 'sim_to_truth')
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Faithful - Error vs Truth
        ax1 = axes[0, 0]
        x = np.arange(2)
        means = [np.mean(f_sim_error), np.mean(f_sim_truth)]
        sems = [stats.sem(f_sim_error), stats.sem(f_sim_truth)]
        colors = ['#E63946', '#457B9D']
        
        bars = ax1.bar(x, means, yerr=sems, color=colors, alpha=0.8, 
                      capsize=6, edgecolor='black', linewidth=1.5)
        ax1.set_xticks(x)
        ax1.set_xticklabels(['To Error', 'To Truth'])
        ax1.set_ylabel('Cosine Similarity', fontsize=11, fontweight='bold')
        ax1.set_title('[B-02] Faithful: Similarity Breakdown', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        for bar, mean, sem in zip(bars, means, sems):
            ax1.text(bar.get_x() + bar.get_width()/2, mean + sem, 
                    f'{mean:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Plot 2: Self-Corrected - Error vs Truth
        ax2 = axes[0, 1]
        means = [np.mean(c_sim_error), np.mean(c_sim_truth)]
        sems = [stats.sem(c_sim_error), stats.sem(c_sim_truth)]
        
        bars = ax2.bar(x, means, yerr=sems, color=colors, alpha=0.8, 
                      capsize=6, edgecolor='black', linewidth=1.5)
        ax2.set_xticks(x)
        ax2.set_xticklabels(['To Error', 'To Truth'])
        ax2.set_ylabel('Cosine Similarity', fontsize=11, fontweight='bold')
        ax2.set_title('[B-02] Self-Corrected: Similarity Breakdown', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        for bar, mean, sem in zip(bars, means, sems):
            ax2.text(bar.get_x() + bar.get_width()/2, mean + sem, 
                    f'{mean:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Plot 3: Compare similarity to error between groups
        ax3 = axes[1, 0]
        parts = ax3.violinplot([f_sim_error, c_sim_error], positions=[0, 1],
                              showmeans=True, showextrema=True, widths=0.6)
        for pc in parts['bodies']:
            pc.set_alpha(0.7)
        
        ax3.set_xticks([0, 1])
        ax3.set_xticklabels(['Faithful', 'Self-Corrected'])
        ax3.set_ylabel('Similarity to Error', fontsize=11, fontweight='bold')
        ax3.set_title('[B-02] Similarity to Error Embedding', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Add stats
        stats_error = self.compute_stats(f_sim_error, c_sim_error)
        if stats_error:
            max_y = max(np.max(f_sim_error), np.max(c_sim_error))
            self.add_significance_stars(ax3, 0, 1, max_y*1.05, stats_error['p_value'])
        
        # Plot 4: Compare similarity to truth between groups
        ax4 = axes[1, 1]
        parts = ax4.violinplot([f_sim_truth, c_sim_truth], positions=[0, 1],
                              showmeans=True, showextrema=True, widths=0.6)
        for pc in parts['bodies']:
            pc.set_alpha(0.7)
        
        ax4.set_xticks([0, 1])
        ax4.set_xticklabels(['Faithful', 'Self-Corrected'])
        ax4.set_ylabel('Similarity to Truth', fontsize=11, fontweight='bold')
        ax4.set_title('[B-02] Similarity to Truth Embedding', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Add stats
        stats_truth = self.compute_stats(f_sim_truth, c_sim_truth)
        if stats_truth:
            max_y = max(np.max(f_sim_truth), np.max(c_sim_truth))
            self.add_significance_stars(ax4, 0, 1, max_y*1.05, stats_truth['p_value'])
        
        plt.tight_layout()
        filename = f"{OUTPUT_DIR}02_detailed_similarity_breakdown.png"
        plt.savefig(filename, dpi=DPI, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {filename}")
    
    def plot_distribution_histograms(self):
        """Plot 3: Overlaid histogram distributions."""
        print(f"{C.OKCYAN}Generating distribution histograms...{C.ENDC}")
        
        f_bias = self.extract_metric(self.faithful, 'belief_bias')
        c_bias = self.extract_metric(self.corrected, 'belief_bias')
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE_DOUBLE)
        
        # Overlaid histograms
        ax1.hist(f_bias, bins=30, alpha=0.6, label='Faithful', color='#E63946', density=True)
        ax1.hist(c_bias, bins=30, alpha=0.6, label='Self-Corrected', color='#06A77D', density=True)
        ax1.axvline(x=0, color='black', linestyle='--', linewidth=2, label='Zero (No Bias)')
        ax1.axvline(x=np.mean(f_bias), color='#E63946', linestyle='-', linewidth=2, alpha=0.8)
        ax1.axvline(x=np.mean(c_bias), color='#06A77D', linestyle='-', linewidth=2, alpha=0.8)
        ax1.set_xlabel('Belief Bias', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Density', fontsize=11, fontweight='bold')
        ax1.set_title('[B-02] Belief Bias Distribution', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Q-Q plot
        from scipy import stats as sp_stats
        theoretical_quantiles = sp_stats.norm.ppf(np.linspace(0.01, 0.99, len(f_bias)))
        f_bias_sorted = np.sort(f_bias)
        
        ax2.scatter(theoretical_quantiles, f_bias_sorted, alpha=0.6, 
                   label='Faithful', color='#E63946', s=30)
        
        theoretical_quantiles_c = sp_stats.norm.ppf(np.linspace(0.01, 0.99, len(c_bias)))
        c_bias_sorted = np.sort(c_bias)
        
        ax2.scatter(theoretical_quantiles_c, c_bias_sorted, alpha=0.6,
                   label='Self-Corrected', color='#06A77D', s=30)
        
        # Reference line
        min_val = min(np.min(f_bias_sorted), np.min(c_bias_sorted))
        max_val = max(np.max(f_bias_sorted), np.max(c_bias_sorted))
        ax2.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Normal')
        
        ax2.set_xlabel('Theoretical Quantiles', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Sample Quantiles', fontsize=11, fontweight='bold')
        ax2.set_title('[B-02] Q-Q Plot (Normality Check)', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = f"{OUTPUT_DIR}03_distribution_histograms.png"
        plt.savefig(filename, dpi=DPI, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {filename}")
    
    def plot_scatter_error_vs_truth(self):
        """Plot 4: Scatter plot of similarity to error vs similarity to truth."""
        print(f"{C.OKCYAN}Generating scatter plot...{C.ENDC}")
        
        f_sim_error = self.extract_metric(self.faithful, 'sim_to_error')
        f_sim_truth = self.extract_metric(self.faithful, 'sim_to_truth')
        c_sim_error = self.extract_metric(self.corrected, 'sim_to_error')
        c_sim_truth = self.extract_metric(self.corrected, 'sim_to_truth')
        
        fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
        
        ax.scatter(f_sim_truth, f_sim_error, alpha=0.5, s=50, 
                  label='Faithful', color='#E63946', edgecolors='black', linewidth=0.5)
        ax.scatter(c_sim_truth, c_sim_error, alpha=0.5, s=50,
                  label='Self-Corrected', color='#06A77D', edgecolors='black', linewidth=0.5)
        
        # Diagonal line (where sim_error = sim_truth)
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),
            np.max([ax.get_xlim(), ax.get_ylim()]),
        ]
        ax.plot(lims, lims, 'k--', alpha=0.5, linewidth=2, label='Equal Similarity')
        
        # Reference lines at 0
        ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
        ax.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
        
        ax.set_xlabel('Similarity to Truth', fontsize=12, fontweight='bold')
        ax.set_ylabel('Similarity to Error', fontsize=12, fontweight='bold')
        ax.set_title('[B-02] Similarity Space: Error vs Truth\nAbove diagonal = More aligned with error', 
                    fontsize=13, fontweight='bold', pad=15)
        ax.legend(fontsize=11, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # Add text annotations for quadrants
        ax.text(0.05, 0.95, 'Both negative\n(Far from both)', 
               transform=ax.transAxes, fontsize=9, va='top', ha='left',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        plt.tight_layout()
        filename = f"{OUTPUT_DIR}04_scatter_error_vs_truth.png"
        plt.savefig(filename, dpi=DPI, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {filename}")
    
    def plot_belief_trajectory(self):
        """Plot 5: Belief bias by problem ID (trajectory plot)."""
        print(f"{C.OKCYAN}Generating belief trajectory plot...{C.ENDC}")
        
        f_ids = [r['problem_id'] for r in self.faithful]
        f_bias = self.extract_metric(self.faithful, 'belief_bias')
        c_ids = [r['problem_id'] for r in self.corrected]
        c_bias = self.extract_metric(self.corrected, 'belief_bias')
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=False)
        
        # Faithful trajectory
        ax1.scatter(f_ids, f_bias, alpha=0.6, s=40, color='#E63946', edgecolors='black', linewidth=0.5)
        ax1.axhline(y=0, color='black', linestyle='--', linewidth=2, alpha=0.7)
        ax1.axhline(y=np.mean(f_bias), color='#E63946', linestyle='-', linewidth=2, alpha=0.8, label=f'Mean = {np.mean(f_bias):.4f}')
        ax1.fill_between(f_ids, np.mean(f_bias) - np.std(f_bias), np.mean(f_bias) + np.std(f_bias), 
                        alpha=0.2, color='#E63946')
        ax1.set_ylabel('Belief Bias', fontsize=11, fontweight='bold')
        ax1.set_title('[B-02] Faithful: Belief Bias by Problem\nPositive = Aligned with Error', 
                     fontsize=12, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Self-corrected trajectory
        ax2.scatter(c_ids, c_bias, alpha=0.6, s=40, color='#06A77D', edgecolors='black', linewidth=0.5)
        ax2.axhline(y=0, color='black', linestyle='--', linewidth=2, alpha=0.7)
        ax2.axhline(y=np.mean(c_bias), color='#06A77D', linestyle='-', linewidth=2, alpha=0.8, label=f'Mean = {np.mean(c_bias):.4f}')
        ax2.fill_between(c_ids, np.mean(c_bias) - np.std(c_bias), np.mean(c_bias) + np.std(c_bias), 
                        alpha=0.2, color='#06A77D')
        ax2.set_xlabel('Problem ID', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Belief Bias', fontsize=11, fontweight='bold')
        ax2.set_title('[B-02] Self-Corrected: Belief Bias by Problem\nPositive = Aligned with Error', 
                     fontsize=12, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = f"{OUTPUT_DIR}05_belief_trajectory.png"
        plt.savefig(filename, dpi=DPI, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {filename}")
    
    def plot_statistical_summary(self):
        """Plot 6: Comprehensive statistical summary."""
        print(f"{C.OKCYAN}Generating statistical summary...{C.ENDC}")
        
        metrics = [
            ('belief_bias', 'Belief Bias\n(error - truth)'),
            ('sim_to_error', 'Similarity\nto Error'),
            ('sim_to_truth', 'Similarity\nto Truth')
        ]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Effect sizes
        ax1 = axes[0, 0]
        effect_sizes = []
        p_values = []
        labels = []
        
        for metric, label in metrics:
            f_data = self.extract_metric(self.faithful, metric)
            c_data = self.extract_metric(self.corrected, metric)
            stats_result = self.compute_stats(f_data, c_data)
            
            if stats_result:
                effect_sizes.append(stats_result['cohens_d'])
                p_values.append(stats_result['p_value'])
                labels.append(label)
        
        colors = ['#d73027' if abs(d) > 0.5 else '#fdae61' if abs(d) > 0.2 else '#91bfdb' 
                  for d in effect_sizes]
        
        y_pos = np.arange(len(labels))
        bars = ax1.barh(y_pos, effect_sizes, color=colors, alpha=0.8, edgecolor='black')
        
        for i, (d, p) in enumerate(zip(effect_sizes, p_values)):
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
            x_pos = d + (0.05 if d > 0 else -0.05)
            ax1.text(x_pos, i, f'{d:.3f}{sig}', va='center', fontsize=9, fontweight='bold')
        
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(labels, fontsize=10)
        ax1.set_xlabel("Cohen's d", fontsize=11, fontweight='bold')
        ax1.set_title("[B-02] Effect Sizes\n(Faithful - Self-Corrected)", fontsize=12, fontweight='bold')
        ax1.axvline(x=0, color='black', linestyle='--', linewidth=1.5)
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Plot 2: P-values (log scale)
        ax2 = axes[0, 1]
        colors_p = ['#006400' if p < 0.001 else '#228B22' if p < 0.01 else '#90EE90' if p < 0.05 else '#FFB6C1'
                    for p in p_values]
        
        bars = ax2.barh(y_pos, [-np.log10(p) for p in p_values], color=colors_p, alpha=0.8, edgecolor='black')
        
        for i, p in enumerate(p_values):
            ax2.text(-np.log10(p) + 0.1, i, f'p={p:.4f}', va='center', fontsize=9, fontweight='bold')
        
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(labels, fontsize=10)
        ax2.set_xlabel('-log10(p-value)', fontsize=11, fontweight='bold')
        ax2.set_title('[B-02] Statistical Significance', fontsize=12, fontweight='bold')
        ax2.axvline(x=-np.log10(0.05), color='red', linestyle='--', linewidth=1.5, label='p=0.05')
        ax2.axvline(x=-np.log10(0.01), color='darkred', linestyle='--', linewidth=1.5, label='p=0.01')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3, axis='x')
        
        # Plot 3: Mean comparison
        ax3 = axes[1, 0]
        x = np.arange(len(metrics))
        width = 0.35
        
        f_means = []
        f_sems = []
        c_means = []
        c_sems = []
        
        for metric, _ in metrics:
            f_data = self.extract_metric(self.faithful, metric)
            c_data = self.extract_metric(self.corrected, metric)
            f_means.append(np.mean(f_data))
            f_sems.append(stats.sem(f_data))
            c_means.append(np.mean(c_data))
            c_sems.append(stats.sem(c_data))
        
        bars1 = ax3.bar(x - width/2, f_means, width, yerr=f_sems,
                       label='Faithful', alpha=0.8, capsize=5, color='#E63946', edgecolor='black')
        bars2 = ax3.bar(x + width/2, c_means, width, yerr=c_sems,
                       label='Self-Corrected', alpha=0.8, capsize=5, color='#06A77D', edgecolor='black')
        
        ax3.set_xticks(x)
        ax3.set_xticklabels([label for _, label in metrics], fontsize=9)
        ax3.set_ylabel('Mean Value', fontsize=11, fontweight='bold')
        ax3.set_title('[B-02] Mean Comparison', fontsize=12, fontweight='bold')
        ax3.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Summary statistics table
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Create table data
        table_data = [
            ['Metric', 'Faithful', 'Self-Corrected', 'p-value'],
        ]
        
        for metric, label in metrics:
            f_data = self.extract_metric(self.faithful, metric)
            c_data = self.extract_metric(self.corrected, metric)
            stats_result = self.compute_stats(f_data, c_data)
            
            f_str = f'{np.mean(f_data):.4f}±{stats.sem(f_data):.4f}'
            c_str = f'{np.mean(c_data):.4f}±{stats.sem(c_data):.4f}'
            p_str = f'{stats_result["p_value"]:.4f}' if stats_result else 'N/A'
            
            table_data.append([label.replace('\n', ' '), f_str, c_str, p_str])
        
        table = ax4.table(cellText=table_data, cellLoc='center', loc='center',
                         colWidths=[0.3, 0.25, 0.25, 0.2])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Style header row
        for i in range(4):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Alternate row colors
        for i in range(1, len(table_data)):
            for j in range(4):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#E7E6E6')
        
        ax4.set_title('[B-02] Summary Statistics', fontsize=12, fontweight='bold', pad=20)
        
        plt.tight_layout()
        filename = f"{OUTPUT_DIR}06_statistical_summary.png"
        plt.savefig(filename, dpi=DPI, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {filename}")
    
    def plot_boxplot_comparison(self):
        """Plot 7: Box plot comparison."""
        print(f"{C.OKCYAN}Generating box plot comparison...{C.ENDC}")
        
        f_bias = self.extract_metric(self.faithful, 'belief_bias')
        c_bias = self.extract_metric(self.corrected, 'belief_bias')
        f_sim_error = self.extract_metric(self.faithful, 'sim_to_error')
        c_sim_error = self.extract_metric(self.corrected, 'sim_to_error')
        f_sim_truth = self.extract_metric(self.faithful, 'sim_to_truth')
        c_sim_truth = self.extract_metric(self.corrected, 'sim_to_truth')
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Belief bias
        ax1 = axes[0]
        bp1 = ax1.boxplot([f_bias, c_bias], labels=['Faithful', 'Self-Corrected'],
                          patch_artist=True, showmeans=True, meanline=True)
        for patch, color in zip(bp1['boxes'], ['#E63946', '#06A77D']):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax1.axhline(y=0, color='black', linestyle='--', linewidth=2, alpha=0.5)
        ax1.set_ylabel('Belief Bias', fontsize=11, fontweight='bold')
        ax1.set_title('[B-02] Belief Bias', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Similarity to error
        ax2 = axes[1]
        bp2 = ax2.boxplot([f_sim_error, c_sim_error], labels=['Faithful', 'Self-Corrected'],
                          patch_artist=True, showmeans=True, meanline=True)
        for patch, color in zip(bp2['boxes'], ['#E63946', '#06A77D']):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax2.axhline(y=0, color='black', linestyle='--', linewidth=2, alpha=0.5)
        ax2.set_ylabel('Similarity to Error', fontsize=11, fontweight='bold')
        ax2.set_title('[B-02] Similarity to Error', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Similarity to truth
        ax3 = axes[2]
        bp3 = ax3.boxplot([f_sim_truth, c_sim_truth], labels=['Faithful', 'Self-Corrected'],
                          patch_artist=True, showmeans=True, meanline=True)
        for patch, color in zip(bp3['boxes'], ['#E63946', '#06A77D']):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax3.axhline(y=0, color='black', linestyle='--', linewidth=2, alpha=0.5)
        ax3.set_ylabel('Similarity to Truth', fontsize=11, fontweight='bold')
        ax3.set_title('[B-02] Similarity to Truth', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        filename = f"{OUTPUT_DIR}07_boxplot_comparison.png"
        plt.savefig(filename, dpi=DPI, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {filename}")
    
    def plot_correlation_analysis(self):
        """Plot 8: Correlation between metrics."""
        print(f"{C.OKCYAN}Generating correlation analysis...{C.ENDC}")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE_DOUBLE)
        
        # Faithful correlations
        f_bias = self.extract_metric(self.faithful, 'belief_bias')
        f_sim_error = self.extract_metric(self.faithful, 'sim_to_error')
        f_sim_truth = self.extract_metric(self.faithful, 'sim_to_truth')
        
        # Ensure same length
        min_len = min(len(f_bias), len(f_sim_error), len(f_sim_truth))
        f_data = np.array([f_bias[:min_len], f_sim_error[:min_len], f_sim_truth[:min_len]])
        f_corr = np.corrcoef(f_data)
        
        labels = ['Belief Bias', 'Sim Error', 'Sim Truth']
        im1 = ax1.imshow(f_corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        ax1.set_xticks(np.arange(len(labels)))
        ax1.set_yticks(np.arange(len(labels)))
        ax1.set_xticklabels(labels, fontsize=10)
        ax1.set_yticklabels(labels, fontsize=10)
        
        # Add correlation values
        for i in range(len(labels)):
            for j in range(len(labels)):
                text = ax1.text(j, i, f'{f_corr[i, j]:.2f}',
                              ha="center", va="center", color="black", fontsize=11, fontweight='bold')
        
        ax1.set_title('[B-02] Faithful: Metric Correlations', fontsize=12, fontweight='bold')
        fig.colorbar(im1, ax=ax1, label='Correlation')
        
        # Self-corrected correlations
        c_bias = self.extract_metric(self.corrected, 'belief_bias')
        c_sim_error = self.extract_metric(self.corrected, 'sim_to_error')
        c_sim_truth = self.extract_metric(self.corrected, 'sim_to_truth')
        
        min_len = min(len(c_bias), len(c_sim_error), len(c_sim_truth))
        c_data = np.array([c_bias[:min_len], c_sim_error[:min_len], c_sim_truth[:min_len]])
        c_corr = np.corrcoef(c_data)
        
        im2 = ax2.imshow(c_corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        ax2.set_xticks(np.arange(len(labels)))
        ax2.set_yticks(np.arange(len(labels)))
        ax2.set_xticklabels(labels, fontsize=10)
        ax2.set_yticklabels(labels, fontsize=10)
        
        for i in range(len(labels)):
            for j in range(len(labels)):
                text = ax2.text(j, i, f'{c_corr[i, j]:.2f}',
                              ha="center", va="center", color="black", fontsize=11, fontweight='bold')
        
        ax2.set_title('[B-02] Self-Corrected: Metric Correlations', fontsize=12, fontweight='bold')
        fig.colorbar(im2, ax=ax2, label='Correlation')
        
        plt.tight_layout()
        filename = f"{OUTPUT_DIR}08_correlation_analysis.png"
        plt.savefig(filename, dpi=DPI, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {filename}")
    
    def generate_all_plots(self):
        """Generate all visualizations."""
        print(f"{C.HEADER}{'='*80}{C.ENDC}")
        print(f"{C.HEADER}Generating All Embedding Visualizations{C.ENDC}")
        print(f"{C.HEADER}{'='*80}{C.ENDC}\n")
        
        self.plot_belief_bias_comparison()
        self.plot_detailed_similarity_breakdown()
        self.plot_distribution_histograms()
        self.plot_scatter_error_vs_truth()
        self.plot_belief_trajectory()
        self.plot_statistical_summary()
        self.plot_boxplot_comparison()
        self.plot_correlation_analysis()
        
        print(f"\n{C.OKGREEN}All visualizations generated successfully!{C.ENDC}")
        print(f"{C.BOLD}Output directory: {OUTPUT_DIR}{C.ENDC}")


def main():
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    logger = Logger(LOG_FILE)
    sys.stdout = logger
    
    try:
        print(f"Embedding Visualization Generation Log")
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}\n")
        
        generator = EmbeddingVisualizationGenerator(INPUT_FILE)
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