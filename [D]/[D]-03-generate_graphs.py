"""
[D]-03 generate_graphs.py:

Phase D: Visualization
Part 03: Pre-Computation Analysis Visualizations

This script generates visualizations for the pre-computation vs live reasoning test,
investigating whether self-correction is due to pre-computed answers.

Visualizations:
1. Probe accuracy comparison (train vs test)
2. Above-baseline performance comparison
3. Accuracy breakdown by split
4. Effect size and statistical summary
5. Hypothesis testing visualization
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
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

INPUT_FILE = "input/[B]-03-output.json"
OUTPUT_DIR = "output/[D]-03-figures/"

# Plot styling
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
FIGSIZE_SINGLE = (10, 6)
FIGSIZE_DOUBLE = (12, 5)
FIGSIZE_LARGE = (14, 8)
DPI = 300


class PrecomputationVisualizationGenerator:
    def __init__(self, results_path):
        print(f"{C.OKCYAN}Loading results from {results_path}...{C.ENDC}")
        with open(results_path, 'r') as f:
            self.results = json.load(f)
        
        self.faithful = self.results['faithful']['probe_results']
        self.corrected = self.results['corrected']['probe_results']
        self.metadata = self.results['metadata']
        
        print(f"{C.OKGREEN}Loaded probe results for analysis{C.ENDC}\n")
    
    def plot_probe_accuracy_comparison(self):
        """Plot 1: Main probe accuracy comparison."""
        print(f"{C.OKCYAN}Generating probe accuracy comparison...{C.ENDC}")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE_DOUBLE)
        
        # Plot 1: Train vs Test accuracy
        categories = ['Train', 'Test', 'Baseline']
        faithful_accs = [
            self.faithful['train_accuracy'],
            self.faithful['test_accuracy'],
            self.faithful['baseline_accuracy']
        ]
        corrected_accs = [
            self.corrected['train_accuracy'],
            self.corrected['test_accuracy'],
            self.corrected['baseline_accuracy']
        ]
        
        x = np.arange(len(categories))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, faithful_accs, width, label='Faithful',
                       alpha=0.8, color='#E63946', edgecolor='black', linewidth=1.5)
        bars2 = ax1.bar(x + width/2, corrected_accs, width, label='Self-Corrected',
                       alpha=0.8, color='#06A77D', edgecolor='black', linewidth=1.5)
        
        ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax1.set_title('[B-03] Linear Probe Accuracy\nPredicting Answer from Early Hidden States',
                     fontsize=13, fontweight='bold', pad=15)
        ax1.set_xticks(x)
        ax1.set_xticklabels(categories, fontsize=11)
        ax1.legend(fontsize=11)
        ax1.set_ylim(0, 1.0)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}',
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Plot 2: Above baseline comparison
        faithful_above = self.faithful['test_accuracy'] - self.faithful['baseline_accuracy']
        corrected_above = self.corrected['test_accuracy'] - self.corrected['baseline_accuracy']
        
        categories2 = ['Faithful', 'Self-Corrected']
        above_baseline = [faithful_above, corrected_above]
        colors = ['#E63946', '#06A77D']
        
        bars = ax2.bar(categories2, above_baseline, color=colors, alpha=0.8,
                      edgecolor='black', linewidth=1.5)
        
        ax2.axhline(y=0, color='black', linestyle='--', linewidth=2, alpha=0.7)
        ax2.set_ylabel('Accuracy Above Baseline', fontsize=12, fontweight='bold')
        ax2.set_title('[B-03] Predictive Power\n(Test - Baseline)',
                     fontsize=13, fontweight='bold', pad=15)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels and interpretation
        for i, (bar, val) in enumerate(zip(bars, above_baseline)):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.3f}',
                    ha='center', va='bottom' if val > 0 else 'top',
                    fontsize=10, fontweight='bold')
            
            # Add interpretation text
            interp = "Pre-computed" if val > 0.1 else "Live reasoning"
            ax2.text(bar.get_x() + bar.get_width()/2., -0.05,
                    interp, ha='center', va='top', fontsize=9, style='italic')
        
        # Add difference annotation
        diff = corrected_above - faithful_above
        mid_x = 0.5
        max_y = max(above_baseline) * 1.1
        ax2.annotate('', xy=(0, max_y), xytext=(1, max_y),
                    arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
        ax2.text(mid_x, max_y, f'Δ = {diff:+.3f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        
        plt.tight_layout()
        filename = f"{OUTPUT_DIR}01_probe_accuracy_comparison.png"
        plt.savefig(filename, dpi=DPI, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {filename}")
    
    def plot_accuracy_breakdown(self):
        """Plot 2: Detailed accuracy breakdown."""
        print(f"{C.OKCYAN}Generating accuracy breakdown...{C.ENDC}")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Faithful - all metrics
        ax1 = axes[0, 0]
        metrics = ['Train', 'Test', 'Baseline']
        values = [
            self.faithful['train_accuracy'],
            self.faithful['test_accuracy'],
            self.faithful['baseline_accuracy']
        ]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        bars = ax1.bar(metrics, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax1.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
        ax1.set_title('[B-03] Faithful: Probe Performance', fontsize=12, fontweight='bold')
        ax1.set_ylim(0, 1.0)
        ax1.grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars, values):
            ax1.text(bar.get_x() + bar.get_width()/2., val,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Add sample size
        ax1.text(0.5, 0.95, f"n_train={self.faithful['n_train']}, n_test={self.faithful['n_test']}",
                transform=ax1.transAxes, ha='center', va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # Plot 2: Self-corrected - all metrics
        ax2 = axes[0, 1]
        values = [
            self.corrected['train_accuracy'],
            self.corrected['test_accuracy'],
            self.corrected['baseline_accuracy']
        ]
        
        bars = ax2.bar(metrics, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax2.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
        ax2.set_title('[B-03] Self-Corrected: Probe Performance', fontsize=12, fontweight='bold')
        ax2.set_ylim(0, 1.0)
        ax2.grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars, values):
            ax2.text(bar.get_x() + bar.get_width()/2., val,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax2.text(0.5, 0.95, f"n_train={self.corrected['n_train']}, n_test={self.corrected['n_test']}",
                transform=ax2.transAxes, ha='center', va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # Plot 3: Test accuracy comparison
        ax3 = axes[1, 0]
        categories = ['Faithful', 'Self-Corrected']
        test_accs = [self.faithful['test_accuracy'], self.corrected['test_accuracy']]
        baseline_accs = [self.faithful['baseline_accuracy'], self.corrected['baseline_accuracy']]
        
        x = np.arange(len(categories))
        width = 0.35
        
        bars1 = ax3.bar(x - width/2, baseline_accs, width, label='Baseline',
                       alpha=0.8, color='#999999', edgecolor='black')
        bars2 = ax3.bar(x + width/2, test_accs, width, label='Test Accuracy',
                       alpha=0.8, color='#E91E63', edgecolor='black')
        
        ax3.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
        ax3.set_title('[B-03] Test Accuracy vs Baseline', fontsize=12, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(categories)
        ax3.legend(fontsize=10)
        ax3.set_ylim(0, 1.0)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Overfitting check (train - test gap)
        ax4 = axes[1, 1]
        faithful_gap = self.faithful['train_accuracy'] - self.faithful['test_accuracy']
        corrected_gap = self.corrected['train_accuracy'] - self.corrected['test_accuracy']
        
        gaps = [faithful_gap, corrected_gap]
        colors_gap = ['#E63946', '#06A77D']
        
        bars = ax4.bar(categories, gaps, color=colors_gap, alpha=0.8,
                      edgecolor='black', linewidth=1.5)
        
        ax4.axhline(y=0, color='black', linestyle='--', linewidth=1.5)
        ax4.set_ylabel('Train - Test Gap', fontsize=11, fontweight='bold')
        ax4.set_title('[B-03] Overfitting Check\n(Lower = Better Generalization)',
                     fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars, gaps):
            ax4.text(bar.get_x() + bar.get_width()/2., val,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        filename = f"{OUTPUT_DIR}02_accuracy_breakdown.png"
        plt.savefig(filename, dpi=DPI, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {filename}")
    
    def plot_hypothesis_test(self):
        """Plot 3: Hypothesis testing visualization."""
        print(f"{C.OKCYAN}Generating hypothesis test visualization...{C.ENDC}")
        
        fig = plt.figure(figsize=FIGSIZE_LARGE)
        gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.3)
        
        # Main hypothesis test
        ax_main = fig.add_subplot(gs[0, :])
        
        faithful_above = self.faithful['test_accuracy'] - self.faithful['baseline_accuracy']
        corrected_above = self.corrected['test_accuracy'] - self.corrected['baseline_accuracy']
        diff = corrected_above - faithful_above
        
        # Create visualization of hypothesis
        categories = ['Faithful', 'Self-Corrected']
        values = [faithful_above, corrected_above]
        colors = ['#E63946', '#06A77D']
        
        bars = ax_main.bar(categories, values, color=colors, alpha=0.8,
                          edgecolor='black', linewidth=2, width=0.5)
        
        ax_main.axhline(y=0, color='black', linestyle='-', linewidth=2)
        ax_main.axhline(y=0.1, color='red', linestyle='--', linewidth=2, alpha=0.5,
                       label='Pre-computation threshold (0.10)')
        
        ax_main.set_ylabel('Accuracy Above Baseline', fontsize=13, fontweight='bold')
        ax_main.set_title('[B-03] HYPOTHESIS TEST: Pre-Computation Detection\n' + 
                         'H₀: Self-corrected models pre-compute answers (>0.10 above baseline)\n' +
                         'H₁: Both groups reason step-by-step (≤0.10 above baseline)',
                         fontsize=14, fontweight='bold', pad=20)
        ax_main.legend(fontsize=11)
        ax_main.grid(True, alpha=0.3, axis='y')
        ax_main.set_ylim(-0.1, 0.3)
        
        # Add values and arrows
        for i, (bar, val) in enumerate(zip(bars, values)):
            height = bar.get_height()
            ax_main.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{val:.4f}', ha='center', va='bottom',
                        fontsize=12, fontweight='bold')
            
            # Arrow to threshold
            if val < 0.1:
                ax_main.annotate('', xy=(bar.get_x() + bar.get_width()/2., 0.1),
                               xytext=(bar.get_x() + bar.get_width()/2., val),
                               arrowprops=dict(arrowstyle='->', color='red', lw=2))
        
        # Verdict box
        if corrected_above > 0.1 and faithful_above <= 0.1:
            verdict = "HYPOTHESIS CONFIRMED"
            verdict_color = '#4CAF50'
            explanation = "Self-corrected models show pre-computation"
        elif corrected_above > faithful_above + 0.05:
            verdict = "PARTIAL SUPPORT"
            verdict_color = '#FFC107'
            explanation = "Self-corrected models show more pre-computation"
        else:
            verdict = "HYPOTHESIS REJECTED"
            verdict_color = '#F44336'
            explanation = "No evidence of differential pre-computation"
        
        ax_main.text(0.5, -0.08, f"{verdict}\n{explanation}",
                    transform=ax_main.transAxes, ha='center', va='top',
                    fontsize=13, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.8', facecolor=verdict_color, 
                             alpha=0.3, edgecolor='black', linewidth=2))
        
        # Supporting evidence plots
        ax1 = fig.add_subplot(gs[1, 0])
        ax2 = fig.add_subplot(gs[1, 1])
        ax3 = fig.add_subplot(gs[2, 0])
        ax4 = fig.add_subplot(gs[2, 1])
        
        # Evidence 1: Effect size
        effect_size = (corrected_above - faithful_above) / np.sqrt(
            ((self.corrected['test_accuracy'] * (1 - self.corrected['test_accuracy'])) +
             (self.faithful['test_accuracy'] * (1 - self.faithful['test_accuracy']))) / 2
        )
        
        ax1.barh(['Effect Size'], [effect_size], color='#9C27B0', alpha=0.8, edgecolor='black')
        ax1.axvline(x=0, color='black', linestyle='-', linewidth=2)
        ax1.axvline(x=0.2, color='gray', linestyle='--', alpha=0.5, label='Small (0.2)')
        ax1.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='Medium (0.5)')
        ax1.set_xlabel('Cohen\'s d', fontsize=10, fontweight='bold')
        ax1.set_title('[B-03] Effect Size', fontsize=11, fontweight='bold')
        ax1.legend(fontsize=8)
        ax1.text(effect_size, 0, f'{effect_size:.3f}', ha='left', va='center',
                fontsize=10, fontweight='bold')
        
        # Evidence 2: Difference magnitude
        ax2.bar(['Difference'], [diff], color='#FF5722', alpha=0.8, edgecolor='black')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=2)
        ax2.axhline(y=0.05, color='red', linestyle='--', alpha=0.5, label='Meaningful (0.05)')
        ax2.set_ylabel('Accuracy Difference', fontsize=10, fontweight='bold')
        ax2.set_title('[B-03] Absolute Difference\n(Corrected - Faithful)',
                     fontsize=11, fontweight='bold')
        ax2.legend(fontsize=8)
        ax2.text(0, diff, f'{diff:+.4f}', ha='center', va='bottom',
                fontsize=10, fontweight='bold')
        
        # Evidence 3: Sample sizes
        ax3.bar(['Faithful', 'Self-Corrected'],
               [self.faithful['n_test'], self.corrected['n_test']],
               color=['#E63946', '#06A77D'], alpha=0.8, edgecolor='black')
        ax3.set_ylabel('Sample Size (Test Set)', fontsize=10, fontweight='bold')
        ax3.set_title('[B-03] Statistical Power', fontsize=11, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Evidence 4: Decision tree
        ax4.axis('off')
        decision_text = "DECISION TREE:\n\n"
        
        if corrected_above > 0.1:
            decision_text += "✓ Self-corrected > 0.10\n"
        else:
            decision_text += "✗ Self-corrected ≤ 0.10\n"
        
        if faithful_above > 0.1:
            decision_text += "✓ Faithful > 0.10\n"
        else:
            decision_text += "✗ Faithful ≤ 0.10\n"
        
        if abs(diff) > 0.05:
            decision_text += "✓ |Difference| > 0.05\n"
        else:
            decision_text += "✗ |Difference| ≤ 0.05\n"
        
        decision_text += f"\n→ {verdict}"
        
        ax4.text(0.5, 0.5, decision_text, transform=ax4.transAxes,
                ha='center', va='center', fontsize=11, family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
        
        plt.suptitle('[B-03] Pre-Computation Hypothesis Test', fontsize=16, fontweight='bold', y=0.98)
        
        filename = f"{OUTPUT_DIR}03_hypothesis_test.png"
        plt.savefig(filename, dpi=DPI, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {filename}")
    
    def plot_interpretation_summary(self):
        """Plot 4: Visual interpretation summary."""
        print(f"{C.OKCYAN}Generating interpretation summary...{C.ENDC}")
        
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('off')
        
        # Title
        title_text = "[B-03] INTERPRETATION SUMMARY\nPre-Computation vs Live Reasoning Test"
        ax.text(0.5, 0.95, title_text, transform=ax.transAxes,
               ha='center', va='top', fontsize=16, fontweight='bold')
        
        # Results box
        faithful_above = self.faithful['test_accuracy'] - self.faithful['baseline_accuracy']
        corrected_above = self.corrected['test_accuracy'] - self.corrected['baseline_accuracy']
        diff = corrected_above - faithful_above
        
        results_text = f"""
RESULTS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Faithful (Propagated Error):
  • Test Accuracy:      {self.faithful['test_accuracy']:.4f}
  • Baseline:           {self.faithful['baseline_accuracy']:.4f}
  • Above Baseline:     {faithful_above:+.4f}
  
Self-Corrected:
  • Test Accuracy:      {self.corrected['test_accuracy']:.4f}
  • Baseline:           {self.corrected['baseline_accuracy']:.4f}
  • Above Baseline:     {corrected_above:+.4f}

Difference (Corrected - Faithful): {diff:+.4f}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        
        ax.text(0.5, 0.75, results_text, transform=ax.transAxes,
               ha='center', va='top', fontsize=11, family='monospace',
               bbox=dict(boxstyle='round,pad=1', facecolor='#E8F5E9', 
                        edgecolor='black', linewidth=2))
        
        # Interpretation
        if abs(diff) < 0.05 and corrected_above < 0.1 and faithful_above < 0.1:
            interpretation = """
INTERPRETATION: No Evidence of Pre-Computation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✗ Both groups show minimal predictive power (~0.10 above baseline)
✗ No meaningful difference between groups (Δ < 0.05)
✗ Early hidden states don't encode final answer

CONCLUSION:
• Self-correction is NOT due to pre-computed answers
• Both groups appear to reason step-by-step
• Behavioral differences must stem from other mechanisms:
  - Attention patterns (see B-01 analysis)
  - Error detection capabilities
  - Working memory utilization

IMPLICATION:
The CoT serves as genuine working memory in BOTH cases.
Self-correction represents runtime error detection, not retrieval.
"""
            box_color = '#FFF9C4'
        
        elif corrected_above > faithful_above + 0.05:
            interpretation = """
INTERPRETATION: Evidence of Differential Pre-Computation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✓ Self-corrected models show higher predictive power
✓ Meaningful difference between groups (Δ > 0.05)
→ Self-corrected models encode more answer information early

CONCLUSION:
• Self-correction may involve retrieving pre-computed knowledge
• Faithful models rely more on sequential computation
• CoT serves different purposes in the two groups

IMPLICATION:
Self-correction could be "post-hoc" - the model already knows
the answer and uses CoT to justify it. Faithful models may
genuinely compute step-by-step.
"""
            box_color = '#E1F5FE'
        
        else:
            interpretation = """
INTERPRETATION: Ambiguous Results
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

~ Some evidence of differences, but not conclusive
~ May need larger sample size or different probe design

CONCLUSION:
• Results don't strongly support either hypothesis
• Further investigation needed
"""
            box_color = '#FFF3E0'
        
        ax.text(0.5, 0.35, interpretation, transform=ax.transAxes,
               ha='center', va='top', fontsize=10, family='monospace',
               bbox=dict(boxstyle='round,pad=1', facecolor=box_color,
                        edgecolor='black', linewidth=2))
        
        filename = f"{OUTPUT_DIR}04_interpretation_summary.png"
        plt.savefig(filename, dpi=DPI, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {filename}")
    
    def plot_accuracy_bars_simple(self):
        """Plot 5: Simple, clear accuracy comparison."""
        print(f"{C.OKCYAN}Generating simple accuracy comparison...{C.ENDC}")
        
        fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
        
        # Data
        categories = ['Faithful\nTest', 'Faithful\nBaseline', 
                     'Self-Corrected\nTest', 'Self-Corrected\nBaseline']
        values = [
            self.faithful['test_accuracy'],
            self.faithful['baseline_accuracy'],
            self.corrected['test_accuracy'],
            self.corrected['baseline_accuracy']
        ]
        colors = ['#E63946', '#FFCCCB', '#06A77D', '#90EE90']
        
        bars = ax.bar(categories, values, color=colors, alpha=0.8,
                     edgecolor='black', linewidth=1.5)
        
        ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax.set_title('[B-03] Linear Probe: Can We Predict Final Answer from Early States?\n' +
                    f'(Extracted at {self.metadata["early_position_fraction"]*100:.0f}% through CoT)',
                    fontsize=13, fontweight='bold', pad=15)
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2., val,
                   f'{val:.3f}', ha='center', va='bottom',
                   fontsize=10, fontweight='bold')
        
        # Add brackets showing above-baseline
        faithful_above = self.faithful['test_accuracy'] - self.faithful['baseline_accuracy']
        corrected_above = self.corrected['test_accuracy'] - self.corrected['baseline_accuracy']
        
        # Faithful bracket
        y1 = self.faithful['baseline_accuracy']
        y2 = self.faithful['test_accuracy']
        ax.plot([0, 0], [y1, y2], 'k-', linewidth=2)
        ax.plot([0, 0.1], [y2, y2], 'k-', linewidth=2)
        ax.text(0.15, (y1+y2)/2, f'+{faithful_above:.3f}', fontsize=10, fontweight='bold')
        
        # Corrected bracket
        y1 = self.corrected['baseline_accuracy']
        y2 = self.corrected['test_accuracy']
        ax.plot([2, 2], [y1, y2], 'k-', linewidth=2)
        ax.plot([2, 2.1], [y2, y2], 'k-', linewidth=2)
        ax.text(2.15, (y1+y2)/2, f'+{corrected_above:.3f}', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        filename = f"{OUTPUT_DIR}05_simple_accuracy_bars.png"
        plt.savefig(filename, dpi=DPI, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {filename}")
    
    def generate_all_plots(self):
        """Generate all visualizations."""
        print(f"{C.HEADER}{'='*80}{C.ENDC}")
        print(f"{C.HEADER}Generating All Pre-Computation Visualizations{C.ENDC}")
        print(f"{C.HEADER}{'='*80}{C.ENDC}\n")
        
        self.plot_probe_accuracy_comparison()
        self.plot_accuracy_breakdown()
        self.plot_hypothesis_test()
        self.plot_interpretation_summary()
        self.plot_accuracy_bars_simple()
        
        print(f"\n{C.OKGREEN}All visualizations generated successfully!{C.ENDC}")
        print(f"{C.BOLD}Output directory: {OUTPUT_DIR}{C.ENDC}")


def main():
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Pre-Computation Visualization Generation")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")
    
    generator = PrecomputationVisualizationGenerator(INPUT_FILE)
    generator.generate_all_plots()
    
    print(f"\n{C.HEADER}{'='*80}{C.ENDC}")
    print(f"{C.BOLD}Summary:{C.ENDC}")
    print(f"  • 5 visualizations generated")
    print(f"  • Output directory: {OUTPUT_DIR}")
    
    faithful_above = generator.faithful['test_accuracy'] - generator.faithful['baseline_accuracy']
    corrected_above = generator.corrected['test_accuracy'] - generator.corrected['baseline_accuracy']
    diff = corrected_above - faithful_above
    
    print(f"\n{C.BOLD}Key Finding:{C.ENDC}")
    print(f"  Faithful above baseline:      {faithful_above:+.4f}")
    print(f"  Self-Corrected above baseline: {corrected_above:+.4f}")
    print(f"  Difference:                    {diff:+.4f}")
    
    if abs(diff) < 0.05:
        print(f"  → {C.OKGREEN}No evidence of differential pre-computation{C.ENDC}")
    elif corrected_above > faithful_above:
        print(f"  → {C.WARNING}Self-corrected shows more pre-computation{C.ENDC}")
    else:
        print(f"  → {C.WARNING}Faithful shows more pre-computation{C.ENDC}")
    
    print(f"{C.HEADER}{'='*80}{C.ENDC}")


if __name__ == "__main__":
    main()