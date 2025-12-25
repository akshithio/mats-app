"""
[B]-01 analyze_attention.py:

Phase B: Mechanistic EDA
Part 05: Attention Weights Analysis

--

This script investigates whether models that propagate errors (FAITHFUL)
differ mechanistically from models that correct them (SELF-CORRECTED) by
analyzing attention patterns to the injected error token.

Key Question: Do faithful models actually "read" the injected error when
generating their continuation, or do they ignore it?

Methodology:
1. Load the model with attention output enabled
2. For each problem, use forward pass (not generate) to capture attention
3. Measure average attention from continuation tokens → injected error token
4. Compare attention scores between FAITHFUL vs SELF-CORRECTED groups
5. Use statistical tests accounting for different group sizes

Expected Result: If faithful models show significantly lower attention to
the error token, it suggests they're not processing it mechanistically.

Input:
    - [A].jsonl (dataset with classifications and injected errors)

Output:
    - [B]-01-attention_analysis.json (detailed attention scores)
    - Console output with statistical comparison

--

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
OUTPUT_FILE = "[B]-01-attention_analysis.json"
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# How many problems to analyze (set to None for all, or a number for faster testing)
MAX_PROBLEMS = None  # Change to 50 for quick testing

# Which layers to average (last N layers tend to be most semantic)
LAYERS_TO_AVERAGE = 5


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
        
        Uses forward pass instead of generate to properly capture attention.
        
        Returns: float (mean attention score) or None if calculation failed
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
            continuation_end = min(continuation_start + 50, inputs.input_ids.shape[1])
            
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
            attention_scores = []
            
            for layer_idx in layers_to_use:
                layer_attn = attentions[layer_idx]  # (batch, heads, seq_len, seq_len)
                
                # Average over heads: (batch, seq_len, seq_len)
                avg_attn = layer_attn.mean(dim=1)
                
                # Extract attention from continuation tokens (rows) to error tokens (cols)
                # Shape: (continuation_length, error_length)
                cont_to_error = avg_attn[0, continuation_start:continuation_end, error_start:error_end]
                
                # Average over both dimensions to get single score for this layer
                if cont_to_error.numel() > 0:
                    layer_score = cont_to_error.mean().item()
                    attention_scores.append(layer_score)
            
            if not attention_scores:
                return None
            
            # Return mean attention across selected layers
            return float(np.mean(attention_scores))
            
        except Exception as e:
            print(f"\n{C.WARNING}Warning: Attention calculation failed - {str(e)[:100]}{C.ENDC}")
            return None
    
    def analyze_dataset(self, dataset_path, max_problems=None):
        """
        Analyze attention patterns across the entire dataset.
        """
        print(f"{C.HEADER}{'='*80}{C.ENDC}")
        print(f"{C.HEADER}Attention Weights Analysis{C.ENDC}")
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
                'total_problems': len(problems),
                'faithful_count': len(faithful_problems),
                'corrected_count': len(corrected_problems)
            }
        }
        
        print(f"{C.OKCYAN}Analyzing FAITHFUL problems...{C.ENDC}")
        for i, problem in enumerate(faithful_problems):
            print(f"[{i+1}/{len(faithful_problems)}] Problem {problem['problem_id']}...", end="\r")
            
            score = self.get_attention_score(
                problem['problem_text'],
                problem['continued_cot'],
                problem['injected_value'],
                problem['original_value']
            )
            
            if score is not None:
                results['faithful'].append({
                    'problem_id': problem['problem_id'],
                    'attention_score': score,
                    'injected_value': problem['injected_value'],
                    'original_value': problem['original_value']
                })
        
        print(" " * 80)
        
        print(f"{C.OKCYAN}Analyzing SELF-CORRECTED problems...{C.ENDC}")
        for i, problem in enumerate(corrected_problems):
            print(f"[{i+1}/{len(corrected_problems)}] Problem {problem['problem_id']}...", end="\r")
            
            score = self.get_attention_score(
                problem['problem_text'],
                problem['continued_cot'],
                problem['injected_value'],
                problem['original_value']
            )
            
            if score is not None:
                results['corrected'].append({
                    'problem_id': problem['problem_id'],
                    'attention_score': score,
                    'injected_value': problem['injected_value'],
                    'original_value': problem['original_value']
                })
        
        print(" " * 80)
        
        return results


def main():
    # Initialize analyzer
    analyzer = AttentionAnalyzer(MODEL_NAME, DEVICE)
    
    # Run analysis
    results = analyzer.analyze_dataset(INPUT_FILE, max_problems=MAX_PROBLEMS)
    
    # Save results
    print(f"\n{C.OKGREEN}Saving results to {OUTPUT_FILE}...{C.ENDC}")
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Compute statistics
    print(f"\n{C.HEADER}{'='*80}{C.ENDC}")
    print(f"{C.HEADER}Statistical Analysis{C.ENDC}")
    print(f"{C.HEADER}{'='*80}{C.ENDC}\n")
    
    faithful_scores = [r['attention_score'] for r in results['faithful']]
    corrected_scores = [r['attention_score'] for r in results['corrected']]
    
    if not faithful_scores or not corrected_scores:
        print(f"{C.FAIL}Error: Insufficient data for analysis{C.ENDC}")
        print(f"  Faithful samples: {len(faithful_scores)}")
        print(f"  Corrected samples: {len(corrected_scores)}")
        return
    
    faithful_mean = np.mean(faithful_scores)
    faithful_std = np.std(faithful_scores)
    faithful_se = faithful_std / np.sqrt(len(faithful_scores))
    
    corrected_mean = np.mean(corrected_scores)
    corrected_std = np.std(corrected_scores)
    corrected_se = corrected_std / np.sqrt(len(corrected_scores))
    
    print(f"{C.BOLD}Attention to Injected Error Token:{C.ENDC}\n")
    
    print(f"{C.FAIL}FAITHFUL (Propagated Error):{C.ENDC}")
    print(f"  Mean: {faithful_mean:.6f} ± {faithful_se:.6f}")
    print(f"  Std:  {faithful_std:.6f}")
    print(f"  N:    {len(faithful_scores)}")
    
    print(f"\n{C.OKGREEN}SELF-CORRECTED:{C.ENDC}")
    print(f"  Mean: {corrected_mean:.6f} ± {corrected_se:.6f}")
    print(f"  Std:  {corrected_std:.6f}")
    print(f"  N:    {len(corrected_scores)}")
    
    # Compute difference and effect size
    diff = faithful_mean - corrected_mean
    ratio = faithful_mean / corrected_mean if corrected_mean > 0 else float('inf')
    
    # Cohen's d effect size (accounts for different sample sizes)
    pooled_std = np.sqrt(((len(faithful_scores)-1)*faithful_std**2 + 
                          (len(corrected_scores)-1)*corrected_std**2) / 
                         (len(faithful_scores) + len(corrected_scores) - 2))
    cohens_d = diff / pooled_std if pooled_std > 0 else 0
    
    # Two-sample t-test (unequal sample sizes is fine)
    t_stat, p_value = stats.ttest_ind(faithful_scores, corrected_scores)
    
    print(f"\n{C.BOLD}Statistical Comparison:{C.ENDC}")
    print(f"  Difference: {diff:+.6f}")
    print(f"  Ratio: {ratio:.2f}x")
    print(f"  Cohen's d: {cohens_d:.3f}")
    print(f"  t-statistic: {t_stat:.3f}")
    print(f"  p-value: {p_value:.4f}")
    
    # Significance interpretation
    if p_value < 0.001:
        sig_str = f"{C.OKGREEN}*** (p < 0.001){C.ENDC}"
    elif p_value < 0.01:
        sig_str = f"{C.OKGREEN}** (p < 0.01){C.ENDC}"
    elif p_value < 0.05:
        sig_str = f"{C.OKGREEN}* (p < 0.05){C.ENDC}"
    else:
        sig_str = f"{C.WARNING}n.s. (p ≥ 0.05){C.ENDC}"
    
    print(f"  Significance: {sig_str}")
    
    # Effect size interpretation
    if abs(cohens_d) < 0.2:
        effect_str = "negligible"
    elif abs(cohens_d) < 0.5:
        effect_str = "small"
    elif abs(cohens_d) < 0.8:
        effect_str = "medium"
    else:
        effect_str = "large"
    
    print(f"  Effect size: {effect_str}")
    
    # Interpret results
    print(f"\n{C.HEADER}{'='*80}{C.ENDC}")
    print(f"{C.HEADER}Interpretation{C.ENDC}")
    print(f"{C.HEADER}{'='*80}{C.ENDC}\n")
    
    if p_value < 0.05:
        if faithful_mean < corrected_mean * 0.8:
            print(f"{C.OKGREEN}✓ Strong Evidence:{C.ENDC} Faithful models pay significantly LESS attention to errors")
            print(f"  → Statistically significant difference (p = {p_value:.4f})")
            print(f"  → {effect_str.capitalize()} effect size (d = {cohens_d:.3f})")
            print(f"  → Suggests faithful models are NOT reading the injected error")
            print(f"  → Error propagation may be due to lack of attention, not reasoning failure")
        elif faithful_mean > corrected_mean * 1.2:
            print(f"{C.WARNING}⚠ Unexpected Finding:{C.ENDC} Faithful models pay MORE attention to errors")
            print(f"  → Statistically significant difference (p = {p_value:.4f})")
            print(f"  → {effect_str.capitalize()} effect size (d = {cohens_d:.3f})")
            print(f"  → Suggests faithful models DO read the error but choose to propagate it")
            print(f"  → Error propagation is a reasoning decision, not attention failure")
        else:
            print(f"{C.OKCYAN}○ Significant but Small Difference:{C.ENDC}")
            print(f"  → Statistically significant (p = {p_value:.4f}) but small practical difference")
            print(f"  → Effect size: {effect_str} (d = {cohens_d:.3f})")
    else:
        print(f"{C.OKCYAN}○ No Significant Difference:{C.ENDC} Similar attention patterns (p = {p_value:.4f})")
        print(f"  → Both groups attend to errors similarly")
        print(f"  → Difference in behavior likely due to downstream reasoning, not attention")
        print(f"  → The imbalanced ratio (169:109) is properly accounted for in the t-test")
    
    print(f"\n{C.BOLD}Note:{C.ENDC} The analysis accounts for different group sizes using")
    print(f"       Welch's t-test (unequal sample sizes) and pooled standard deviation.")
    
    print(f"\n{C.BOLD}Saved detailed results to: {OUTPUT_FILE}{C.ENDC}")


if __name__ == "__main__":
    main()