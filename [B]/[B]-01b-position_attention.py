"""
[B]-02 analyze_attention.py:

Phase B: Mechanistic EDA
Part 05: Attention Weights Analysis (Position-Aware)

--

This script investigates whether models that propagate errors (FAITHFUL)
differ mechanistically from models that correct them (SELF-CORRECTED) by
analyzing attention patterns to the injected error token.

Key Question: Do faithful models actually "read" the injected error when
generating their continuation, or do they ignore it?

Methodology:
1. Load the model with attention output enabled
2. For each problem, use forward pass to capture attention
3. Measure attention from continuation tokens → injected error token
4. Analyze BOTH early tokens (computational) and late tokens (summary)
5. Compare attention scores between FAITHFUL vs SELF-CORRECTED groups
6. Use statistical tests accounting for different group sizes

Expected Result: If faithful models show significantly lower attention to
the error token during EARLY continuation (where computation happens), it 
suggests they're not processing it mechanistically.

Input:
    - [A].jsonl (dataset with classifications and injected errors)

Output:
    - [B]-02-position_attention.json (detailed attention scores)
    - Console output with statistical comparison
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import numpy as np
from collections import defaultdict
import re
from scipy import stats

class C:
    HEADER = '\033[95m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# Configuration
MODEL_NAME = "Qwen/Qwen2.5-Math-7B-Instruct"
INPUT_FILE = "[A].jsonl"
OUTPUT_FILE = "[B]-02-position_attention.json"
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# How many problems to analyze (set to None for all, or a number for faster testing)
MAX_PROBLEMS = None  # Change to 50 for quick testing

# Which layers to average (last N layers tend to be most semantic)
LAYERS_TO_AVERAGE = 5

# Position-aware analysis parameters
EARLY_TOKENS = 10  # First N tokens after error (computational phase)
LATE_START = 20    # Start position for late tokens (summary phase)
LATE_TOKENS = 30   # Number of late tokens to analyze


class AttentionAnalyzer:
    def __init__(self, model_name, device):
        print(f"{C.OKCYAN}Loading model: {model_name} on {device}...{C.ENDC}")
        self.device = device
        
        # Load model with attention output enabled
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if device == "mps" else (torch.float16 if device == "cuda" else torch.float32),
            device_map="auto" if device == "cuda" else None,
            attn_implementation="eager"  # Ensures attention weights are accessible
        )
        
        if device in ["mps", "cpu"]:
            self.model.to(device)
        
        self.model.eval()  # Set to evaluation mode
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"{C.OKGREEN}Model loaded successfully!{C.ENDC}\n")
    
    def find_injected_value_in_tokens(self, full_text, injected_value, tokenizer):
        """
        Find token positions where injected value appears.
        Returns list of token indices.
        """
        # Tokenize the full text
        tokens = tokenizer.encode(full_text, add_special_tokens=False)
        
        # Tokenize the injected value
        injected_tokens = tokenizer.encode(str(injected_value), add_special_tokens=False)
        
        # Find positions where injected value appears
        positions = []
        for i in range(len(tokens) - len(injected_tokens) + 1):
            if tokens[i:i+len(injected_tokens)] == injected_tokens:
                positions.extend(range(i, i + len(injected_tokens)))
        
        return positions
    
    def get_attention_score(self, problem_text, full_continuation, 
                           injected_value, original_value):
        """
        Calculate attention score from continuation tokens to injected error token.
        
        Uses forward pass to replay attention patterns from the original generation.
        Now returns BOTH early and late attention scores for position-aware analysis.
        
        Returns: dict with 'overall', 'early', 'late', 'weighted' scores, or None
        """
        try:
            # Construct the full prompt
            messages = [
                {"role": "system", "content": "You are a helpful assistant. Solve math problems step by step."},
                {"role": "user", "content": f"Solve this problem step by step:\n\n{problem_text}"}
            ]
            base_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            # Full sequence is: base_prompt + full_continuation
            full_sequence = base_prompt + full_continuation
            
            # Tokenize
            inputs = self.tokenizer(
                full_sequence, 
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(self.device)
            
            # Find where the injected value is in the token sequence
            error_token_positions = self.find_injected_value_in_tokens(
                full_continuation, 
                injected_value, 
                self.tokenizer
            )
            
            if not error_token_positions:
                return None
            
            # Adjust positions to account for base_prompt
            base_prompt_length = len(self.tokenizer.encode(base_prompt, add_special_tokens=False))
            error_positions = [pos + base_prompt_length for pos in error_token_positions]
            
            # Take first occurrence
            error_start = error_positions[0]
            error_end = error_positions[-1] + 1
            
            # Define "continuation" as tokens after the error
            if error_end >= inputs.input_ids.shape[1] - 5:
                return None  # Not enough continuation after error
            
            continuation_start = error_end
            max_continuation_len = inputs.input_ids.shape[1] - continuation_start
            
            # Check if we have enough tokens for position-aware analysis
            if max_continuation_len < EARLY_TOKENS:
                return None
            
            # Forward pass with attention output
            with torch.no_grad():
                outputs = self.model(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    output_attentions=True,
                    return_dict=True
                )
            
            attentions = outputs.attentions  # Tuple of (num_layers,) each (batch, heads, seq_len, seq_len)
            
            # Select last N layers
            num_layers = len(attentions)
            layers_to_use = list(range(max(0, num_layers - LAYERS_TO_AVERAGE), num_layers))
            
            # Calculate attention from continuation tokens to error tokens
            # Now we'll compute multiple metrics
            
            early_scores = []
            late_scores = []
            all_scores = []
            
            for layer_idx in layers_to_use:
                layer_attn = attentions[layer_idx]  # (batch, heads, seq_len, seq_len)
                
                # Average over heads: (batch, seq_len, seq_len)
                avg_attn = layer_attn.mean(dim=1)
                
                # Extract attention from continuation tokens (rows) to error tokens (cols)
                # Shape: (continuation_length, error_length)
                early_end = min(continuation_start + EARLY_TOKENS, inputs.input_ids.shape[1])
                early_cont_to_error = avg_attn[0, continuation_start:early_end, error_start:error_end]
                
                if early_cont_to_error.numel() > 0:
                    early_scores.append(early_cont_to_error.float().mean().item())
                
                # Late tokens (if available)
                late_start_pos = continuation_start + LATE_START
                late_end_pos = min(late_start_pos + LATE_TOKENS, inputs.input_ids.shape[1])
                
                if late_start_pos < inputs.input_ids.shape[1]:
                    late_cont_to_error = avg_attn[0, late_start_pos:late_end_pos, error_start:error_end]
                    
                    if late_cont_to_error.numel() > 0:
                        late_scores.append(late_cont_to_error.float().mean().item())
                
                # Overall (for comparison with original method)
                continuation_end = min(continuation_start + 50, inputs.input_ids.shape[1])
                cont_to_error = avg_attn[0, continuation_start:continuation_end, error_start:error_end]
                
                if cont_to_error.numel() > 0:
                    all_scores.append(cont_to_error.float().mean().item())
            
            if not early_scores:
                return None
            
            # Compute weighted score (exponential decay favoring early tokens)
            weighted_score = None
            if all_scores:
                # Get the full attention matrix for weighted calculation
                layer_attn = attentions[layers_to_use[-1]]  # Use last layer
                avg_attn = layer_attn.mean(dim=1)
                
                continuation_end = min(continuation_start + 50, inputs.input_ids.shape[1])
                cont_to_error = avg_attn[0, continuation_start:continuation_end, error_start:error_end]
                
                if cont_to_error.numel() > 0:
                    # Create exponential decay weights
                    num_tokens = cont_to_error.shape[0]
                    weights = np.exp(-np.arange(num_tokens) / 10)
                    weights = weights / weights.sum()  # Normalize
                    
                    # Apply weights (mean over error tokens first, then weight over continuation tokens)
                    token_scores = cont_to_error.float().mean(dim=1).cpu().numpy()
                    weighted_score = float(np.sum(token_scores * weights))
            
            return {
                'overall': float(np.mean(all_scores)),
                'early': float(np.mean(early_scores)),
                'late': float(np.mean(late_scores)) if late_scores else None,
                'weighted': weighted_score,
                'num_early_tokens': len(early_scores) * EARLY_TOKENS // len(layers_to_use),
                'num_late_tokens': len(late_scores) * LATE_TOKENS // len(layers_to_use) if late_scores else 0
            }
            
        except Exception as e:
            print(f"\n{C.WARNING}Warning: Attention calculation failed - {str(e)[:100]}{C.ENDC}")
            return None
    
    def analyze_dataset(self, dataset_path, max_problems=None):
        """
        Analyze attention patterns across the entire dataset.
        """
        print(f"{C.HEADER}{'='*80}{C.ENDC}")
        print(f"{C.HEADER}Attention Weights Analysis (Position-Aware){C.ENDC}")
        print(f"{C.HEADER}{'='*80}{C.ENDC}\n")
        
        # Load dataset
        print(f"Loading dataset from {dataset_path}...")
        problems = []
        with open(dataset_path, 'r') as f:
            for line in f:
                if line.strip():
                    problems.append(json.loads(line))
        
        if max_problems:
            problems = problems[:max_problems]
        
        print(f"Loaded {len(problems)} problems\n")
        
        # Separate by classification
        faithful_problems = [p for p in problems if p['classification'] is True]
        corrected_problems = [p for p in problems if p['classification'] is False]
        
        print(f"{C.BOLD}Dataset Composition:{C.ENDC}")
        print(f"  {C.FAIL}Faithful (Propagated Error):{C.ENDC} {len(faithful_problems)}")
        print(f"  {C.OKGREEN}Self-Corrected:{C.ENDC} {len(corrected_problems)}")
        print(f"  Ratio: {len(faithful_problems)/len(corrected_problems):.2f}:1")
        print()
        
        # Analyze each group
        results = {
            'faithful': [],
            'corrected': [],
            'metadata': {
                'model': MODEL_NAME,
                'layers_averaged': LAYERS_TO_AVERAGE,
                'early_tokens': EARLY_TOKENS,
                'late_start': LATE_START,
                'late_tokens': LATE_TOKENS,
                'total_problems': len(problems),
                'faithful_count': len(faithful_problems),
                'corrected_count': len(corrected_problems)
            }
        }
        
        print(f"{C.OKCYAN}Analyzing FAITHFUL problems...{C.ENDC}")
        for i, problem in enumerate(faithful_problems):
            print(f"[{i+1}/{len(faithful_problems)}] Problem {problem['problem_id']}...", end="\r")
            
            scores = self.get_attention_score(
                problem['problem_text'],
                problem['continued_cot'],
                problem['injected_value'],
                problem['original_value']
            )
            
            if scores is not None:
                results['faithful'].append({
                    'problem_id': problem['problem_id'],
                    'attention_overall': scores['overall'],
                    'attention_early': scores['early'],
                    'attention_late': scores['late'],
                    'attention_weighted': scores['weighted'],
                    'injected_value': problem['injected_value'],
                    'original_value': problem['original_value']
                })
        
        print(" " * 80)
        
        print(f"{C.OKCYAN}Analyzing SELF-CORRECTED problems...{C.ENDC}")
        for i, problem in enumerate(corrected_problems):
            print(f"[{i+1}/{len(corrected_problems)}] Problem {problem['problem_id']}...", end="\r")
            
            scores = self.get_attention_score(
                problem['problem_text'],
                problem['continued_cot'],
                problem['injected_value'],
                problem['original_value']
            )
            
            if scores is not None:
                results['corrected'].append({
                    'problem_id': problem['problem_id'],
                    'attention_overall': scores['overall'],
                    'attention_early': scores['early'],
                    'attention_late': scores['late'],
                    'attention_weighted': scores['weighted'],
                    'injected_value': problem['injected_value'],
                    'original_value': problem['original_value']
                })
        
        print(" " * 80)
        
        return results


def print_metric_comparison(metric_name, faithful_scores, corrected_scores, color_code):
    """Helper function to print statistical comparison for a metric."""
    if not faithful_scores or not corrected_scores:
        return
    
    # Filter out None values
    faithful_scores = [s for s in faithful_scores if s is not None]
    corrected_scores = [s for s in corrected_scores if s is not None]
    
    if not faithful_scores or not corrected_scores:
        return
    
    faithful_mean = np.mean(faithful_scores)
    faithful_std = np.std(faithful_scores)
    faithful_se = faithful_std / np.sqrt(len(faithful_scores))
    
    corrected_mean = np.mean(corrected_scores)
    corrected_std = np.std(corrected_scores)
    corrected_se = corrected_std / np.sqrt(len(corrected_scores))
    
    print(f"\n{C.BOLD}{metric_name}:{C.ENDC}\n")
    
    print(f"{C.FAIL}FAITHFUL:{C.ENDC} {faithful_mean:.6f} ± {faithful_se:.6f} (n={len(faithful_scores)})")
    print(f"{C.OKGREEN}CORRECTED:{C.ENDC} {corrected_mean:.6f} ± {corrected_se:.6f} (n={len(corrected_scores)})")
    
    # Statistical test
    diff = faithful_mean - corrected_mean
    ratio = faithful_mean / corrected_mean if corrected_mean > 0 else float('inf')
    
    pooled_std = np.sqrt(((len(faithful_scores)-1)*faithful_std**2 + 
                          (len(corrected_scores)-1)*corrected_std**2) / 
                         (len(faithful_scores) + len(corrected_scores) - 2))
    cohens_d = diff / pooled_std if pooled_std > 0 else 0
    
    t_stat, p_value = stats.ttest_ind(faithful_scores, corrected_scores)
    
    if p_value < 0.001:
        sig_str = "***"
    elif p_value < 0.01:
        sig_str = "**"
    elif p_value < 0.05:
        sig_str = "*"
    else:
        sig_str = "n.s."
    
    print(f"Δ = {diff:+.6f}, ratio = {ratio:.2f}x, d = {cohens_d:.3f}, p = {p_value:.4f} {sig_str}")


def main():
    # Initialize analyzer
    analyzer = AttentionAnalyzer(MODEL_NAME, DEVICE)
    
    # Run analysis
    results = analyzer.analyze_dataset(INPUT_FILE, max_problems=MAX_PROBLEMS)
    
    # Save results
    print(f"\n{C.OKGREEN}Saving results to {OUTPUT_FILE}...{C.ENDC}")
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Compute statistics for all metrics
    print(f"\n{C.HEADER}{'='*80}{C.ENDC}")
    print(f"{C.HEADER}Statistical Analysis - Position-Aware Comparison{C.ENDC}")
    print(f"{C.HEADER}{'='*80}{C.ENDC}")
    
    # Extract scores for each metric
    faithful_overall = [r['attention_overall'] for r in results['faithful']]
    corrected_overall = [r['attention_overall'] for r in results['corrected']]
    
    faithful_early = [r['attention_early'] for r in results['faithful']]
    corrected_early = [r['attention_early'] for r in results['corrected']]
    
    faithful_late = [r['attention_late'] for r in results['faithful'] if r['attention_late'] is not None]
    corrected_late = [r['attention_late'] for r in results['corrected'] if r['attention_late'] is not None]
    
    faithful_weighted = [r['attention_weighted'] for r in results['faithful'] if r['attention_weighted'] is not None]
    corrected_weighted = [r['attention_weighted'] for r in results['corrected'] if r['attention_weighted'] is not None]
    
    # Print comparisons
    print_metric_comparison("OVERALL Attention (Original Method)", faithful_overall, corrected_overall, C.OKCYAN)
    print_metric_comparison("EARLY Attention (First 10 tokens - Computational)", faithful_early, corrected_early, C.OKGREEN)
    print_metric_comparison("LATE Attention (Tokens 20-50 - Summary)", faithful_late, corrected_late, C.WARNING)
    print_metric_comparison("WEIGHTED Attention (Exponential Decay)", faithful_weighted, corrected_weighted, C.OKCYAN)
    
    # Key interpretation
    print(f"\n{C.HEADER}{'='*80}{C.ENDC}")
    print(f"{C.HEADER}Key Findings{C.ENDC}")
    print(f"{C.HEADER}{'='*80}{C.ENDC}\n")
    
    if faithful_early and corrected_early:
        early_faithful_mean = np.mean(faithful_early)
        early_corrected_mean = np.mean(corrected_early)
        _, early_p = stats.ttest_ind(faithful_early, corrected_early)
        
        if early_p < 0.05:
            if early_faithful_mean < early_corrected_mean * 0.8:
                print(f"{C.OKGREEN}✓ EARLY tokens show LOWER attention in faithful models{C.ENDC}")
                print(f"  → Faithful models ignore errors during computational phase")
                print(f"  → This suggests lack of mechanistic processing of the error")
            elif early_faithful_mean > early_corrected_mean * 1.2:
                print(f"{C.WARNING}⚠ EARLY tokens show HIGHER attention in faithful models{C.ENDC}")
                print(f"  → Faithful models attend to errors but propagate anyway")
                print(f"  → This suggests reasoning/decision failure, not attention failure")
            else:
                print(f"{C.OKCYAN}○ EARLY tokens show similar attention{C.ENDC}")
        else:
            print(f"{C.OKCYAN}○ No significant difference in early attention (p = {early_p:.4f}){C.ENDC}")
    
    if faithful_late and corrected_late and len(faithful_late) > 10 and len(corrected_late) > 10:
        _, late_p = stats.ttest_ind(faithful_late, corrected_late)
        if late_p < 0.05:
            print(f"\n{C.BOLD}Note:{C.ENDC} Late tokens (summary phase) show different attention patterns")
            print(f"       This may reflect citation behavior rather than computational use")
    
    print(f"\n{C.BOLD}Position-aware analysis complete!{C.ENDC}")
    print(f"Saved detailed results to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()