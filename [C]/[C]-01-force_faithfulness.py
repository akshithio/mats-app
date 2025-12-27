"""
[C]-01 force_faithfulness.py:

Phase C: Forceful Perturbations
Part 01: Forcing Faithfulness via Specific Head Ablation

--

Research Question 3: Can we force a model to behave faithfully?

Hypothesis:
Self-Correction requires specific "Reset Heads" to look back at the 
Start-of-Sequence (BOS) to re-ground the logic. If we ablate these heads, 
the model cannot verify the truth and must accept the injected error.

Methodology:
1. PHASE 1 (Detection): Run self-correcting examples. Record which Attention Heads 
   attend most strongly to the BOS token during generation.
2. PHASE 2 (Intervention): Take "Self-Corrected" examples (which normally fix the error). 
   Ablate the Top-20 "Reset Heads".
3. SUCCESS METRIC: Faithfulness rate increases (Model stops correcting).

Input: [A].jsonl
Output: [C]-01-output.json, [C]-01-logs.txt
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import re
import sys
import numpy as np
from datetime import datetime
import os

class C:
    HEADER = '\033[95m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

MODEL_NAME = "Qwen/Qwen2.5-Math-7B-Instruct"
INPUT_FILE = "[A].jsonl"
OUTPUT_FILE = "[C]-01-output.json"
LOG_FILE = "[C]-01-logs.txt"
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# Experiment Settings
TOP_K_HEADS = 20
NUM_DETECTION_SAMPLES = 20
NUM_INTERVENTION_SAMPLES = 50


class TeeOutput:
    """Redirect stdout to both console and file"""
    def __init__(self, file_path, original_stream):
        self.file = open(file_path, 'w', encoding='utf-8')
        self.original_stream = original_stream
        
    def write(self, message):
        # Strip ANSI color codes for file output
        clean_message = re.sub(r'\033\[[0-9;]+m', '', message)
        self.file.write(clean_message)
        self.file.flush()
        self.original_stream.write(message)
        self.original_stream.flush()
    
    def flush(self):
        self.file.flush()
        self.original_stream.flush()
    
    def close(self):
        self.file.close()


def extract_answer(text):
    """Extract numerical answer from text."""
    match = re.search(r'####\s*([\d,]+(?:\.\d+)?)', text)
    if match: return float(match.group(1).replace(',', ''))
    # Fallback patterns
    patterns = [
        r'boxed\{([\d\.]+)\}',
        r'is\s+([\d\.]+)\.',
        r'=\s*([\d\.]+)',
    ]
    for p in patterns:
        m = re.search(p, text)
        if m: return float(m.group(1))
    return None


class HeadAblationExperiment:
    def __init__(self):
        print(f"{C.OKGREEN}Loading model: {MODEL_NAME} on {DEVICE}...{C.ENDC}")
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, 
            dtype=torch.bfloat16 if DEVICE == "mps" else torch.float16,
            device_map="auto" if DEVICE == "cuda" else None,
            attn_implementation="eager" # Critical for hooking weights/outputs
        )
        if DEVICE in ["mps", "cpu"]: self.model.to(DEVICE)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        
        self.num_layers = self.model.config.num_hidden_layers
        self.num_heads = self.model.config.num_attention_heads
        self.head_dim = self.model.config.hidden_size // self.num_heads
        
        print(f"Model loaded: {self.num_layers} layers, {self.num_heads} heads.")

    def detect_reset_heads(self, problems):
        """Find heads that attend to BOS (token 0) during the correction phase."""
        print(f"\n{C.HEADER}PHASE 1: Detecting Reset Heads (Scanning {len(problems)} samples)...{C.ENDC}")
        
        # Matrix to store BOS attention [Layers, Heads]
        bos_scores = torch.zeros((self.num_layers, self.num_heads)).to(DEVICE)
        
        for i, prob in enumerate(problems):
            # We want to analyze the attention *after* the injection
            full_text = prob['continued_cot']
            inputs = self.tokenizer(full_text, return_tensors="pt").to(DEVICE)
            
            with torch.no_grad():
                outputs = self.model(**inputs, output_attentions=True)
            
            # outputs.attentions is a tuple of (Batch, Heads, Seq, Seq) for each layer
            # We want to look at the attention from the *last generated tokens* to the *first token*
            
            # Heuristic: Look at the last 20 tokens (likely where correction reasoning happens)
            seq_len = inputs.input_ids.shape[1]
            start_idx = max(0, seq_len - 50) 
            
            for layer_idx, layer_attn in enumerate(outputs.attentions):
                # layer_attn: [1, Heads, Seq, Seq]
                # Extract attention from [start_idx:] to Token 0
                
                # Shape: [Heads, Recent_Tokens]
                attn_to_bos = layer_attn[0, :, start_idx:, 0] 
                
                # Average across the time steps
                avg_score = attn_to_bos.mean(dim=1) # [Heads]
                bos_scores[layer_idx] += avg_score
            
            print(f"Scanning {i+1}/{len(problems)}...", end="\r")

        # Normalize
        bos_scores /= len(problems)
        
        # Get Top K
        flat_indices = torch.topk(bos_scores.flatten(), TOP_K_HEADS).indices
        
        heads_to_ablate = []
        print(f"\n\n{C.BOLD}Top {TOP_K_HEADS} Reset Heads identified:{C.ENDC}")
        for idx in flat_indices:
            l = (idx // self.num_heads).item()
            h = (idx % self.num_heads).item()
            score = bos_scores[l, h].item()
            heads_to_ablate.append((l, h))
            print(f"  Layer {l}, Head {h} (Score: {score:.4f})")
            
        return heads_to_ablate

    def run_intervention(self, problems, heads_to_ablate):
        """Run generation with the identified heads zeroed out."""
        print(f"\n{C.HEADER}PHASE 2: Lobotomy (Intervention on {len(problems)} samples)...{C.ENDC}")
        
        # Prepare Hooks
        hooks = []
        
        def get_ablation_hook(heads_in_layer):
            def hook(module, args, output):
                # output is tuple (hidden_states, ...)
                # hidden_states: [Batch, Seq, Hidden]
                hidden = output[0]
                
                # Zero out specific heads
                # Qwen concatenates heads in the last dimension
                for h in heads_in_layer:
                    start = h * self.head_dim
                    end = (h + 1) * self.head_dim
                    hidden[:, :, start:end] = 0.0
                
                return (hidden,) + output[1:]
            return hook

        # Organize heads by layer
        layer_map = {}
        for l, h in heads_to_ablate:
            if l not in layer_map: layer_map[l] = []
            layer_map[l].append(h)
            
        # Register hooks
        for l, heads in layer_map.items():
            layer_module = self.model.model.layers[l].self_attn
            h = layer_module.register_forward_hook(get_ablation_hook(heads))
            hooks.append(h)
            
        print(f"{C.WARNING}Hooks registered. Generating...{C.ENDC}\n")
        
        results = []
        forced_faithfulness_count = 0
        
        try:
            for i, prob in enumerate(problems):
                # 1. Construct Prompt: Question + CoT up to Injection
                # We need to cut off the 'continued_cot' right after the error
                full_text = prob['continued_cot']
                inj_val = str(prob['injected_value'])
                
                # Find the injection
                # We use the text *up to* the injection + the injection itself
                # But we have to be careful not to include the *correction* that follows
                
                # Heuristic: Find injection, take context up to it + 10 chars (e.g., closing bracket)
                idx = full_text.find(inj_val)
                if idx == -1: 
                    print(f"Skipping {i}: Injection not found in text")
                    continue
                
                # Cutoff point: End of the injected number
                cutoff = idx + len(inj_val)
                
                # To be safe, let's include the line or delimiter if present
                # Looking at your JSON: "\\]\n   So, James"
                # It seems the error is usually inside a math block or sentence
                # Let's feed the text up to the error and let it continue
                
                context_prefix = full_text[:cutoff]
                
                # Re-tokenize
                # We simply create a completion request
                # We need to format it as a user/assistant turn if possible, or raw completion
                # Since we are continuing a thought, raw completion is best
                
                # But Qwen is Chat. We should reconstruct the prompt.
                messages = [
                    {"role": "system", "content": "You are a helpful assistant. Solve math problems step by step."},
                    {"role": "user", "content": f"Solve this problem step by step:\n\n{prob['problem_text']}"}
                ]
                base_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                
                # Find where the CoT starts in the full text (it's the whole continued_cot usually)
                # We will append the context_prefix to the base_prompt
                # Be careful not to duplicate if continued_cot includes prompt (usually it doesn't in dataset output)
                
                input_text = base_prompt + context_prefix
                inputs = self.tokenizer(input_text, return_tensors="pt").to(DEVICE)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=256,
                        do_sample=False,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                # Decode
                generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                new_content = generated[len(base_prompt):] # Extract just the assistant response
                
                # Check Answer
                final_ans = extract_answer(new_content)
                gt = prob['ground_truth_answer']
                
                is_faithful = True # Default assumption
                
                # If it matches GT, it Self-Corrected (Failed to force faithfulness)
                if final_ans is not None and gt is not None:
                    if abs(final_ans - gt) < 0.1:
                        is_faithful = False
                
                # Visual Logging
                status = f"{C.OKGREEN}FORCED FAITHFUL{C.ENDC}" if is_faithful else f"{C.FAIL}STILL CORRECTED{C.ENDC}"
                print(f"[{i+1}/{len(problems)}] GT:{gt} | Gen:{final_ans} | {status}")
                
                if is_faithful: forced_faithfulness_count += 1
                
                results.append({
                    "problem_id": prob['problem_id'],
                    "is_faithful_after_ablation": is_faithful,
                    "generated_text": new_content
                })
                
        finally:
            # Remove hooks
            for h in hooks: h.remove()
            
        return forced_faithfulness_count, len(results), results


def main():
    original_stdout = sys.stdout
    tee_stdout = TeeOutput(LOG_FILE, original_stdout)
    sys.stdout = tee_stdout
    
    try:
        print(f"Execution started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Log file: {LOG_FILE}\n")
        
        # Load JSONL
        print(f"Reading {INPUT_FILE}...")
        data = []
        with open(INPUT_FILE, 'r') as f:
            for line in f:
                if line.strip(): data.append(json.loads(line))
                
        # Filter for SELF-CORRECTED (classification == false)
        # The user noted 'classification' is boolean. False usually means 'Self-Corrected' in this dataset context 
        # (based on previous logs: "is_faithful is False -> RESULT: SELF-CORRECTED")
        
        corrected_problems = [d for d in data if d.get('classification') is False]
        
        print(f"Found {len(corrected_problems)} Self-Corrected examples.")
        if len(corrected_problems) < 10:
            print("Not enough examples to run.")
            return

        exp = HeadAblationExperiment()
        
        # 1. Phase 1: Detection
        detect_set = corrected_problems[:NUM_DETECTION_SAMPLES]
        reset_heads = exp.detect_reset_heads(detect_set)
        
        # 2. Phase 2: Intervention
        # Use a different set if possible, or same if limited data
        intervention_set = corrected_problems[NUM_DETECTION_SAMPLES:NUM_DETECTION_SAMPLES+NUM_INTERVENTION_SAMPLES]
        if len(intervention_set) < 10: intervention_set = corrected_problems[:NUM_INTERVENTION_SAMPLES]
        
        success_count, total, results = exp.run_intervention(intervention_set, reset_heads)
        
        # Save
        print(f"\n{C.OKGREEN}Saving results to {OUTPUT_FILE}...{C.ENDC}")
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"\n{C.HEADER}{'='*60}{C.ENDC}")
        print(f"{C.HEADER}FINAL RESULTS{C.ENDC}")
        print(f"{C.HEADER}{'='*60}{C.ENDC}")
        print(f"Original Behavior: 100% Self-Correcting")
        print(f"After Lobotomy:    {success_count}/{total} became FAITHFUL")
        print(f"Success Rate:      {success_count/total:.1%}")
        print(f"Output saved to:   {OUTPUT_FILE}")
        
        print(f"\n{C.OKGREEN}Analysis complete!{C.ENDC}")
        print(f"Execution completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    finally:
        sys.stdout = original_stdout
        tee_stdout.close()


if __name__ == "__main__":
    main()