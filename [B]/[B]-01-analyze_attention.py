"""
[B]-01 analyze_attention.py:

Phase B: Mechanistic EDA - COMPREHENSIVE VERSION
Part 05: Complete Attention & Residual Stream Analysis

This script combines position-aware temporal analysis with residual stream
investigation to fully understand the mechanistic differences between
faithful and self-corrected models.

Analyses:
1. Backward attention with temporal granularity (early/late/weighted)
2. Error token's contextual attention (what error attends to)
3. Residual stream activation analysis
4. Attention entropy at error position
5. Complete diagnostics and position verification
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import numpy as np
import re
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

# Configuration
MODEL_NAME = "Qwen/Qwen2.5-Math-7B-Instruct"
INPUT_FILE = "[A].jsonl"
OUTPUT_FILE = "[B]-01-output.json"
LOG_FILE = "[B]-01-logs.txt"
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

MAX_PROBLEMS = None
LAYERS_TO_AVERAGE = 5

# Temporal analysis parameters
EARLY_TOKENS = 10  # First N tokens after error (computational phase)
LATE_START = 20    # Start position for late tokens (summary phase)
LATE_TOKENS = 30   # Number of late tokens to analyze


class Logger:
    """Logs output to both console and file, stripping color codes from file."""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8')
        
    def write(self, message):
        self.terminal.write(message)
        clean_message = re.sub(r'\033\[[0-9;]+m', '', message)
        self.log.write(clean_message)
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()
        
    def close(self):
        self.log.close()


class ComprehensiveAnalyzer:
    def __init__(self, model_name, device):
        print(f"{C.OKCYAN}Loading model: {model_name} on {device}...{C.ENDC}")
        self.device = device
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if device == "mps" else (torch.float16 if device == "cuda" else torch.float32),
            device_map="auto" if device == "cuda" else None,
            attn_implementation="eager",
            output_hidden_states=True
        )
        
        if device in ["mps", "cpu"]:
            self.model.to(device)
        
        self.model.eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"{C.OKGREEN}Model loaded successfully!{C.ENDC}\n")
    
    def find_injected_value_in_tokens(self, full_text, injected_value, tokenizer):
        """Find token positions where injected value appears."""
        tokens = tokenizer.encode(full_text, add_special_tokens=False)
        injected_tokens = tokenizer.encode(str(injected_value), add_special_tokens=False)
        
        positions = []
        for i in range(len(tokens) - len(injected_tokens) + 1):
            if tokens[i:i+len(injected_tokens)] == injected_tokens:
                positions.extend(range(i, i + len(injected_tokens)))
        
        return positions
    
    def get_comprehensive_analysis(self, problem_text, full_continuation, 
                                   injected_value, original_value, verbose=False):
        """
        Complete mechanistic analysis including:
        - Temporal backward attention (early/late/weighted)
        - Error contextual attention
        - Residual stream effects
        - Attention entropy
        """
        try:
            # Construct full prompt
            messages = [
                {"role": "system", "content": "You are a helpful assistant. Solve math problems step by step."},
                {"role": "user", "content": f"Solve this problem step by step:\n\n{problem_text}"}
            ]
            base_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            full_sequence = base_prompt + full_continuation
            
            # Tokenize
            inputs = self.tokenizer(
                full_sequence, 
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(self.device)
            
            # Find error token positions
            error_token_positions = self.find_injected_value_in_tokens(
                full_continuation, 
                injected_value, 
                self.tokenizer
            )
            
            if not error_token_positions:
                return None
            
            # Adjust for base prompt
            base_prompt_length = len(self.tokenizer.encode(base_prompt, add_special_tokens=False))
            error_positions = [pos + base_prompt_length for pos in error_token_positions]
            
            error_start = error_positions[0]
            error_end = error_positions[-1] + 1
            
            # Define continuation (must be after error)
            if error_end >= inputs.input_ids.shape[1] - 5:
                return None
            
            continuation_start = error_end
            continuation_end = min(continuation_start + 50, inputs.input_ids.shape[1])
            
            # Check for minimum continuation length
            if continuation_end - continuation_start < EARLY_TOKENS:
                return None
            
            # Diagnostic output
            if verbose:
                print(f"\n{C.OKCYAN}Token Position Diagnostics:{C.ENDC}")
                print(f"  Error tokens [{error_start}:{error_end}]: '{self.tokenizer.decode(inputs.input_ids[0, error_start:error_end])}'")
                print(f"  Continuation [{continuation_start}:{continuation_end}]: '{self.tokenizer.decode(inputs.input_ids[0, continuation_start:continuation_end])[:100]}...'")
                print(f"  Total sequence length: {inputs.input_ids.shape[1]}")
            
            # Forward pass with attention and hidden states
            with torch.no_grad():
                outputs = self.model(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    output_attentions=True,
                    output_hidden_states=True,
                    return_dict=True
                )
            
            attentions = outputs.attentions
            hidden_states = outputs.hidden_states
            
            # Select last N layers for analysis
            num_layers = len(attentions)
            layers_to_use = list(range(max(0, num_layers - LAYERS_TO_AVERAGE), num_layers))
            
            # ===== TEMPORAL BACKWARD ATTENTION ANALYSIS =====
            overall_scores = []
            early_scores = []
            late_scores = []
            first_token_scores = []
            last_token_scores = []
            
            for layer_idx in layers_to_use:
                layer_attn = attentions[layer_idx]
                avg_attn = layer_attn.mean(dim=1)  # Average over heads
                
                # Overall backward attention
                cont_to_error = avg_attn[0, continuation_start:continuation_end, error_start:error_end]
                if cont_to_error.numel() > 0:
                    overall_scores.append(cont_to_error.float().mean().item())
                
                # Early tokens (computational phase)
                early_end = min(continuation_start + EARLY_TOKENS, continuation_end)
                early_to_error = avg_attn[0, continuation_start:early_end, error_start:error_end]
                if early_to_error.numel() > 0:
                    early_scores.append(early_to_error.float().mean().item())
                
                # Late tokens (summary phase)
                late_start_pos = continuation_start + LATE_START
                late_end_pos = min(late_start_pos + LATE_TOKENS, continuation_end)
                if late_start_pos < continuation_end:
                    late_to_error = avg_attn[0, late_start_pos:late_end_pos, error_start:error_end]
                    if late_to_error.numel() > 0:
                        late_scores.append(late_to_error.float().mean().item())
                
                # First and last token granularity
                first_attn = avg_attn[0, continuation_start, error_start:error_end]
                if first_attn.numel() > 0:
                    first_token_scores.append(first_attn.float().mean().item())
                
                last_attn = avg_attn[0, continuation_end-1, error_start:error_end]
                if last_attn.numel() > 0:
                    last_token_scores.append(last_attn.float().mean().item())
            
            backward_overall = float(np.mean(overall_scores)) if overall_scores else None
            backward_early = float(np.mean(early_scores)) if early_scores else None
            backward_late = float(np.mean(late_scores)) if late_scores else None
            first_token_attention = float(np.mean(first_token_scores)) if first_token_scores else None
            last_token_attention = float(np.mean(last_token_scores)) if last_token_scores else None
            
            # Weighted attention (exponential decay)
            weighted_score = None
            if overall_scores:
                layer_attn = attentions[layers_to_use[-1]]
                avg_attn = layer_attn.mean(dim=1)
                cont_to_error = avg_attn[0, continuation_start:continuation_end, error_start:error_end]
                
                if cont_to_error.numel() > 0:
                    num_tokens = cont_to_error.shape[0]
                    weights = np.exp(-np.arange(num_tokens) / 10)
                    weights = weights / weights.sum()
                    token_scores = cont_to_error.float().mean(dim=1).cpu().numpy()
                    weighted_score = float(np.sum(token_scores * weights))
            
            # ===== ERROR CONTEXTUAL ATTENTION =====
            error_context_scores = []
            for layer_idx in layers_to_use:
                layer_attn = attentions[layer_idx]
                avg_attn = layer_attn.mean(dim=1)
                error_to_context = avg_attn[0, error_start:error_end, :error_start]
                if error_to_context.numel() > 0:
                    error_context_scores.append(error_to_context.float().mean().item())
            
            error_context_attention = float(np.mean(error_context_scores)) if error_context_scores else None
            
            # ===== RESIDUAL STREAM ANALYSIS =====
            residual_effects = []
            for layer_idx in layers_to_use:
                hidden = hidden_states[layer_idx + 1]
                
                error_hidden = hidden[0, error_start:error_end, :]
                error_norm = torch.norm(error_hidden, dim=-1).mean().item()
                
                if error_start > 10:
                    context_hidden = hidden[0, error_start-10:error_start, :]
                    context_norm = torch.norm(context_hidden, dim=-1).mean().item()
                    
                    if context_norm > 0:
                        residual_effects.append(error_norm / context_norm)
            
            residual_stream_ratio = float(np.mean(residual_effects)) if residual_effects else None
            
            # ===== ATTENTION ENTROPY AT ERROR =====
            error_entropy_scores = []
            for layer_idx in layers_to_use:
                layer_attn = attentions[layer_idx]
                avg_attn = layer_attn.mean(dim=1)
                error_attn = avg_attn[0, error_start:error_end, :error_start]
                
                if error_attn.numel() > 0:
                    attn_probs = error_attn.flatten()
                    attn_probs = attn_probs[attn_probs > 1e-10]
                    if attn_probs.numel() > 0:
                        entropy = -(attn_probs * torch.log(attn_probs)).sum().item()
                        error_entropy_scores.append(entropy)
            
            error_attention_entropy = float(np.mean(error_entropy_scores)) if error_entropy_scores else None
            
            return {
                # Temporal backward attention
                'backward_overall': backward_overall,
                'backward_early': backward_early,
                'backward_late': backward_late,
                'backward_weighted': weighted_score,
                'first_token_attention': first_token_attention,
                'last_token_attention': last_token_attention,
                # Forward contextual attention
                'error_context_attention': error_context_attention,
                # Residual stream
                'residual_stream_ratio': residual_stream_ratio,
                # Entropy
                'error_attention_entropy': error_attention_entropy,
                # Metadata
                'continuation_length': continuation_end - continuation_start,
                'error_length': error_end - error_start
            }
            
        except Exception as e:
            print(f"\n{C.WARNING}Warning: Analysis failed - {str(e)[:150]}{C.ENDC}")
            return None
    
    def analyze_dataset(self, dataset_path, max_problems=None):
        """Analyze attention patterns across the entire dataset."""
        print(f"{C.HEADER}{'='*80}{C.ENDC}")
        print(f"{C.HEADER}Comprehensive Mechanistic Analysis{C.ENDC}")
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
        
        # Analyze faithful problems
        print(f"{C.OKCYAN}Analyzing FAITHFUL problems...{C.ENDC}")
        for i, problem in enumerate(faithful_problems):
            print(f"[{i+1}/{len(faithful_problems)}] Problem {problem['problem_id']}...", end="\r")
            
            analysis = self.get_comprehensive_analysis(
                problem['problem_text'],
                problem['continued_cot'],
                problem['injected_value'],
                problem['original_value'],
                verbose=(i == 0)
            )
            
            if analysis is not None:
                results['faithful'].append({
                    'problem_id': problem['problem_id'],
                    **analysis,
                    'injected_value': problem['injected_value'],
                    'original_value': problem['original_value']
                })
        
        print(" " * 80)
        
        # Analyze corrected problems
        print(f"{C.OKCYAN}Analyzing SELF-CORRECTED problems...{C.ENDC}")
        for i, problem in enumerate(corrected_problems):
            print(f"[{i+1}/{len(corrected_problems)}] Problem {problem['problem_id']}...", end="\r")
            
            analysis = self.get_comprehensive_analysis(
                problem['problem_text'],
                problem['continued_cot'],
                problem['injected_value'],
                problem['original_value'],
                verbose=(i == 0)
            )
            
            if analysis is not None:
                results['corrected'].append({
                    'problem_id': problem['problem_id'],
                    **analysis,
                    'injected_value': problem['injected_value'],
                    'original_value': problem['original_value']
                })
        
        print(" " * 80)
        
        return results


def compute_statistics(faithful_data, corrected_data, metric_name):
    """Compute statistics for a single metric."""
    # Filter out None values
    faithful_data = [x for x in faithful_data if x is not None]
    corrected_data = [x for x in corrected_data if x is not None]
    
    if not faithful_data or not corrected_data:
        return None
    
    f_mean = np.mean(faithful_data)
    f_std = np.std(faithful_data)
    f_se = f_std / np.sqrt(len(faithful_data))
    
    c_mean = np.mean(corrected_data)
    c_std = np.std(corrected_data)
    c_se = c_std / np.sqrt(len(corrected_data))
    
    # Statistical test
    t_stat, p_value = stats.ttest_ind(faithful_data, corrected_data)
    
    # Effect size
    pooled_std = np.sqrt(((len(faithful_data)-1)*f_std**2 + 
                          (len(corrected_data)-1)*c_std**2) / 
                         (len(faithful_data) + len(corrected_data) - 2))
    cohens_d = (f_mean - c_mean) / pooled_std if pooled_std > 0 else 0
    
    return {
        'metric': metric_name,
        'faithful_mean': f_mean,
        'faithful_std': f_std,
        'faithful_se': f_se,
        'faithful_n': len(faithful_data),
        'corrected_mean': c_mean,
        'corrected_std': c_std,
        'corrected_se': c_se,
        'corrected_n': len(corrected_data),
        'difference': f_mean - c_mean,
        'ratio': f_mean / c_mean if c_mean != 0 else float('inf'),
        't_statistic': t_stat,
        'p_value': p_value,
        'cohens_d': cohens_d
    }


def print_metric_stats(stats):
    """Pretty print statistics for one metric."""
    print(f"\n{C.BOLD}{stats['metric']}:{C.ENDC}")
    print(f"  {C.FAIL}Faithful:{C.ENDC} {stats['faithful_mean']:.6f} ± {stats['faithful_se']:.6f} (n={stats['faithful_n']})")
    print(f"  {C.OKGREEN}Corrected:{C.ENDC} {stats['corrected_mean']:.6f} ± {stats['corrected_se']:.6f} (n={stats['corrected_n']})")
    print(f"  Δ: {stats['difference']:+.6f}, Ratio: {stats['ratio']:.2f}x, d: {stats['cohens_d']:.3f}, p: {stats['p_value']:.4f}", end="")
    
    if stats['p_value'] < 0.001:
        print(f" {C.OKGREEN}***{C.ENDC}")
    elif stats['p_value'] < 0.01:
        print(f" {C.OKGREEN}**{C.ENDC}")
    elif stats['p_value'] < 0.05:
        print(f" {C.OKGREEN}*{C.ENDC}")
    else:
        print(f" {C.WARNING}n.s.{C.ENDC}")


def main():
    logger = Logger(LOG_FILE)
    sys.stdout = logger
    
    try:
        print(f"Comprehensive Mechanistic Analysis Log")
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}\n")
        
        # Initialize analyzer
        analyzer = ComprehensiveAnalyzer(MODEL_NAME, DEVICE)
        
        # Run analysis
        results = analyzer.analyze_dataset(INPUT_FILE, max_problems=MAX_PROBLEMS)
        
        # Save results
        print(f"\n{C.OKGREEN}Saving results to {OUTPUT_FILE}...{C.ENDC}")
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Statistical Analysis
        print(f"\n{C.HEADER}{'='*80}{C.ENDC}")
        print(f"{C.HEADER}Statistical Analysis - All Metrics{C.ENDC}")
        print(f"{C.HEADER}{'='*80}{C.ENDC}")
        
        metrics = [
            ('backward_overall', 'Backward Attention (Overall)'),
            ('backward_early', 'Backward Attention (Early - Computational Phase)'),
            ('backward_late', 'Backward Attention (Late - Summary Phase)'),
            ('backward_weighted', 'Backward Attention (Weighted - Decay)'),
            ('first_token_attention', 'First Token Attention'),
            ('last_token_attention', 'Last Token Attention'),
            ('error_context_attention', 'Error Context Attention (Forward)'),
            ('residual_stream_ratio', 'Residual Stream Ratio'),
            ('error_attention_entropy', 'Error Attention Entropy')
        ]
        
        all_stats = {}
        
        for metric_key, metric_name in metrics:
            faithful_values = [r[metric_key] for r in results['faithful'] if r.get(metric_key) is not None]
            corrected_values = [r[metric_key] for r in results['corrected'] if r.get(metric_key) is not None]
            
            if faithful_values and corrected_values:
                stats_result = compute_statistics(faithful_values, corrected_values, metric_name)
                all_stats[metric_key] = stats_result
                print_metric_stats(stats_result)
        
        # === INTERPRETATION ===
        print(f"\n{C.HEADER}{'='*80}{C.ENDC}")
        print(f"{C.HEADER}Key Findings & Interpretation{C.ENDC}")
        print(f"{C.HEADER}{'='*80}{C.ENDC}\n")
        
        # 1. Early vs Late temporal analysis
        if 'backward_early' in all_stats and 'backward_late' in all_stats:
            early = all_stats['backward_early']
            late = all_stats['backward_late']
            
            print(f"{C.BOLD}1. Temporal Pattern (Early vs Late Attention):{C.ENDC}")
            print(f"   Early (computational): F={early['faithful_mean']:.6f}, C={early['corrected_mean']:.6f}")
            print(f"   Late (summary): F={late['faithful_mean']:.6f}, C={late['corrected_mean']:.6f}")
            
            if early['p_value'] < 0.05:
                if early['difference'] < 0:
                    print(f"   {C.OKGREEN}✓{C.ENDC} Faithful models show LOWER early attention")
                    print(f"   → They ignore errors during computation")
                else:
                    print(f"   {C.WARNING}○{C.ENDC} Faithful models show HIGHER early attention")
                    print(f"   → They read errors but propagate anyway")
            
            # Check decay pattern
            f_decay = early['faithful_mean'] - late['faithful_mean']
            c_decay = early['corrected_mean'] - late['corrected_mean']
            print(f"\n   Attention decay: F={f_decay:+.6f}, C={c_decay:+.6f}")
        
        # 2. Error contextual integration
        if 'error_context_attention' in all_stats:
            eca = all_stats['error_context_attention']
            print(f"\n{C.BOLD}2. Error Context Attention (Error → Context):{C.ENDC}")
            if eca['p_value'] < 0.05:
                if eca['difference'] > 0:
                    print(f"   {C.OKGREEN}✓{C.ENDC} Faithful errors attend MORE to context (p={eca['p_value']:.3f})")
                    print(f"   → Error gets integrated into representation")
                else:
                    print(f"   {C.WARNING}○{C.ENDC} Corrected errors attend MORE to context")
            else:
                print(f"   {C.OKCYAN}○{C.ENDC} Similar contextual attention (p={eca['p_value']:.3f})")
        
        # 3. CRITICAL: Residual stream
        if 'residual_stream_ratio' in all_stats:
            rsr = all_stats['residual_stream_ratio']
            print(f"\n{C.BOLD}3. Residual Stream Effects (CRITICAL FOR PARADOX):{C.ENDC}")
            if rsr['p_value'] < 0.05:
                if rsr['difference'] > 0:
                    print(f"   {C.OKGREEN}✓✓✓ PARADOX RESOLVED!{C.ENDC}")
                    print(f"   → Faithful errors have STRONGER residual presence")
                    print(f"   → F={rsr['faithful_mean']:.3f} vs C={rsr['corrected_mean']:.3f} (p={rsr['p_value']:.4f})")
                    print(f"   → Effect size: {rsr['cohens_d']:.3f}")
                    print(f"\n   {C.BOLD}Explanation:{C.ENDC}")
                    print(f"   • Low backward attention = not actively inspecting")
                    print(f"   • High residual ratio = error baked into representation")
                    print(f"   • Propagation via residual stream, NOT attention!")
                else:
                    print(f"   {C.WARNING}○ Unexpected: Corrected errors stronger in residual{C.ENDC}")
            else:
                print(f"   {C.OKCYAN}○ No significant residual difference (p={rsr['p_value']:.3f}){C.ENDC}")
                print(f"   → Paradox remains unresolved")
        
        # 4. Entropy analysis
        if 'error_attention_entropy' in all_stats:
            entropy = all_stats['error_attention_entropy']
            print(f"\n{C.BOLD}4. Attention Entropy at Error:{C.ENDC}")
            if entropy['p_value'] < 0.05:
                if entropy['difference'] > 0:
                    print(f"   {C.OKGREEN}✓{C.ENDC} Faithful errors have HIGHER entropy")
                    print(f"   → More diffuse attention = better integration")
                else:
                    print(f"   {C.OKGREEN}✓{C.ENDC} Faithful errors have LOWER entropy")
                    print(f"   → More focused attention = selective integration")
        
        # === SUMMARY ===
        print(f"\n{C.HEADER}{'='*80}{C.ENDC}")
        print(f"{C.HEADER}SUMMARY: The Paradox Explained{C.ENDC}")
        print(f"{C.HEADER}{'='*80}{C.ENDC}\n")
        
        print(f"{C.BOLD}Original Paradox:{C.ENDC}")
        print(f"  • Intervention study: Faithful models have HIGH causal dependence on errors")
        print(f"  • Attention study: Faithful models have LOW backward attention to errors")
        print(f"  • How can they depend on something they don't look at?\n")
        
        if 'residual_stream_ratio' in all_stats and all_stats['residual_stream_ratio']['p_value'] < 0.05:
            rsr = all_stats['residual_stream_ratio']
            if rsr['difference'] > 0:
                print(f"{C.OKGREEN}✓ RESOLUTION CONFIRMED:{C.ENDC}")
                print(f"  1. Errors leave strong residual stream signatures")
                print(f"  2. Future tokens inherit these corrupted representations")
                print(f"  3. No explicit backward attention needed")
                print(f"  4. Propagation is IMPLICIT through the residual stream\n")
                print(f"{C.BOLD}Mechanism:{C.ENDC}")
                print(f"  • Faithful models: Error → Residual Stream → Passive Inheritance")
                print(f"  • Self-corrected models: Error → Inspection (high attention) → Rejection\n")
                print(f"{C.BOLD}This explains:{C.ENDC}")
                print(f"  ✓ Why intervention has large effect (breaks residual stream)")
                print(f"  ✓ Why attention is low (no need to look back)")
                print(f"  ✓ Why early attention matters (sets up residual corruption)")
            else:
                print(f"{C.WARNING}○ PARTIAL RESOLUTION:{C.ENDC}")
                print(f"  • Residual stream shows opposite pattern")
                print(f"  • May need deeper investigation (activation patching, etc.)")
        else:
            print(f"{C.WARNING}○ PARADOX UNRESOLVED:{C.ENDC}")
            print(f"  • No significant residual stream difference")
            print(f"  • Alternative hypotheses to explore:")
            print(f"    1. Different layer analysis (early vs late layers)")
            print(f"    2. Activation patching to trace information flow")
            print(f"    3. Value/key/query decomposition")
            print(f"    4. MLP activations (bypassing attention entirely)")
        
        print(f"\n{C.HEADER}{'='*80}{C.ENDC}")
        print(f"{C.BOLD}Files Saved:{C.ENDC}")
        print(f"  • {OUTPUT_FILE} (complete metrics)")
        print(f"  • {LOG_FILE} (this analysis)")
        print(f"\n{C.BOLD}Next Steps:{C.ENDC}")
        if 'residual_stream_ratio' in all_stats and all_stats['residual_stream_ratio']['p_value'] < 0.05:
            print(f"  ✓ Mechanism identified! Consider:")
            print(f"    - Activation patching to confirm causal path")
            print(f"    - Layer-wise analysis to find critical layer")
            print(f"    - Probing for error representations")
        else:
            print(f"  → Need deeper mechanistic investigation:")
            print(f"    - Analyze MLP activations (not just attention)")
            print(f"    - Try different layer ranges")
            print(f"    - Consider model-specific architectures")
        
        print(f"\n{C.OKGREEN}Analysis complete!{C.ENDC}")
        print(f"Execution time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    finally:
        sys.stdout = logger.terminal
        logger.close()
        print(f"\n{C.OKGREEN}Full output saved to {LOG_FILE}{C.ENDC}")


if __name__ == "__main__":
    main()