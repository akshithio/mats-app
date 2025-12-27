"""
[C]-02 activation_steering.py

Phase C: Forceful Perturbations
Part 02: Forcing Faithfulness via Activation Steering (The "Sedative" Vector)

--

Research Question 3: Can we force a model to behave faithfully?

Hypothesis:
Self-Correction is triggered by a "Loud" error signal. Faithfulness happens when 
the signal is "Quiet". By injecting the vector difference (Faithful - Corrected), 
we can "sedate" the model's alarm system, dampening the error signal and forcing 
it to accept the injection.

Methodology:
1. HARVEST: Run 50 Faithful and 50 Corrected examples. Capture residual streams 
   at the exact token position of the injected error.
2. COMPUTE: Calculate `Steering_Vector = Mean(Faithful) - Mean(Corrected)` for each layer.
3. INTERVENE: Run "Self-Correcting" examples (which normally reject the error). 
   Add `coeff * Steering_Vector` to the residual stream at the error position.
4. METRIC: Success if the model stops correcting and propagates the error.

Input: [A].jsonl
Output: [C]-02-output.json, [C]-02-logs.txt
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import re
import sys
from datetime import datetime

class C:
    HEADER = '\033[95m'
    OKGREEN = '\033[92m'
    OKCYAN = '\033[96m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

MODEL_NAME = "Qwen/Qwen2.5-Math-7B-Instruct"
INPUT_FILE = "[A].jsonl"
OUTPUT_FILE = "[C]-02-output.json"
LOG_FILE = "[C]-02-logs.txt"
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# Experiment Settings
NUM_HARVEST_SAMPLES = 50   # How many samples to build the vector
NUM_TEST_SAMPLES = 50      # How many to test the intervention on
STEERING_COEFF = 1.5       # Strength of the injection
INJECTION_LAYERS = range(10, 26) # Middle-Late layers usually handle logic/checks


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
    match = re.search(r'####\s*([\d,]+(?:\.\d+)?)', text)
    if match: return float(match.group(1).replace(',', ''))
    patterns = [r'boxed\{([\d\.]+)\}', r'is\s+([\d\.]+)\.', r'=\s*([\d\.]+)']
    for p in patterns:
        m = re.search(p, text)
        if m: return float(m.group(1))
    return None


class SteeringExperiment:
    def __init__(self):
        print(f"{C.OKGREEN}Loading model: {MODEL_NAME} on {DEVICE}...{C.ENDC}")
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, 
            dtype=torch.bfloat16 if DEVICE == "mps" else torch.float16,
            device_map="auto" if DEVICE == "cuda" else None,
        )
        if DEVICE in ["mps", "cpu"]: self.model.to(DEVICE)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.num_layers = self.model.config.num_hidden_layers
        self.hidden_size = self.model.config.hidden_size
        print(f"Model loaded: {self.num_layers} layers.")

    def get_hidden_states_at_error(self, problems):
        """Capture hidden states at the last token of the injected error."""
        layer_states = {l: [] for l in range(self.num_layers)}
        
        for i, prob in enumerate(problems):
            # Construct text up to the error
            full_text = prob['continued_cot']
            inj_val = str(prob['injected_value'])
            
            # Find position
            idx = full_text.find(inj_val)
            if idx == -1: continue
            
            # We want the state right AFTER processing the error token(s)
            cutoff = idx + len(inj_val)
            prefix = full_text[:cutoff]
            
            inputs = self.tokenizer(prefix, return_tensors="pt").to(DEVICE)
            
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
            
            # Outputs.hidden_states is tuple of (Batch, Seq, Hidden)
            # Take the last token (the error token)
            for l in range(self.num_layers):
                # Layer 0 output is at index 1 (index 0 is embeddings)
                state = outputs.hidden_states[l+1][0, -1, :].cpu()
                layer_states[l].append(state)
            
            print(f"Harvesting {i+1}/{len(problems)}...", end="\r")
            
        return layer_states

    def compute_steering_vector(self, faithful_data, corrected_data):
        print(f"\n{C.HEADER}PHASE 1: Harvesting Vectors...{C.ENDC}")
        
        print(f"  Harvesting FAITHFUL states...")
        faithful_states = self.get_hidden_states_at_error(faithful_data)
        
        print(f"\n  Harvesting CORRECTED states...")
        corrected_states = self.get_hidden_states_at_error(corrected_data)
        
        steering_vectors = {}
        print(f"\n  Computing Mean Difference (Faithful - Corrected)...")
        
        for l in range(self.num_layers):
            if not faithful_states[l] or not corrected_states[l]:
                continue
                
            f_stack = torch.stack(faithful_states[l])
            c_stack = torch.stack(corrected_states[l])
            
            f_mean = f_stack.mean(dim=0)
            c_mean = c_stack.mean(dim=0)
            
            # Vector: Moves from Corrected -> Faithful
            # Concept: "Make it look more like a Faithful state"
            vec = f_mean - c_mean
            
            steering_vectors[l] = vec.to(DEVICE)
            
        print(f"  Computed {len(steering_vectors)} steering vectors.")
        return steering_vectors

    def run_intervention(self, problems, vectors):
        print(f"\n{C.HEADER}PHASE 2: Steering Intervention (The Sedative)...{C.ENDC}")
        print(f"Injecting vector into Layers {min(INJECTION_LAYERS)}-{max(INJECTION_LAYERS)} with coeff {STEERING_COEFF}")
        
        results = []
        success_count = 0
        
        # --- FIXED HOOK FUNCTION ---
        def get_steering_hook(layer_idx, vec):
            def hook(module, args, output):
                # Handle Tuple vs Tensor output
                if isinstance(output, tuple):
                    hidden = output[0]
                    # Ensure dtype matches
                    v = vec.to(dtype=hidden.dtype, device=hidden.device)
                    # Add vector (broadcasting across sequence)
                    hidden = hidden + (v * STEERING_COEFF)
                    return (hidden,) + output[1:]
                else:
                    # Output is just the Tensor
                    v = vec.to(dtype=output.dtype, device=output.device)
                    return output + (v * STEERING_COEFF)
            return hook
        # ---------------------------

        # Register Hooks
        hooks = []
        for l in INJECTION_LAYERS:
            if l in vectors:
                layer_module = self.model.model.layers[l]
                h = layer_module.register_forward_hook(get_steering_hook(l, vectors[l]))
                hooks.append(h)
                
        print(f"{C.WARNING}Steering active. {len(hooks)} hooks registered. Generating...{C.ENDC}\n")
        
        try:
            for i, prob in enumerate(problems):
                # Reconstruct Prompt
                full_text = prob['continued_cot']
                inj_val = str(prob['injected_value'])
                idx = full_text.find(inj_val)
                if idx == -1: 
                    print(f"Skipping {i}: Injection not found in text")
                    continue
                
                # Context up to error
                cutoff = idx + len(inj_val)
                context_prefix = full_text[:cutoff]
                
                messages = [
                    {"role": "system", "content": "You are a helpful assistant. Solve math problems step by step."},
                    {"role": "user", "content": f"Solve this problem step by step:\n\n{prob['problem_text']}"}
                ]
                base_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                input_text = base_prompt + context_prefix
                inputs = self.tokenizer(input_text, return_tensors="pt").to(DEVICE)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=256,
                        do_sample=False,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                new_content = generated[len(base_prompt):] 
                
                # Check Answer
                final_ans = extract_answer(new_content)
                gt = prob['ground_truth_answer']
                
                is_faithful = True
                if final_ans is not None and gt is not None:
                    if abs(final_ans - gt) < 0.1:
                        is_faithful = False
                
                status = f"{C.OKGREEN}FORCED FAITHFUL{C.ENDC}" if is_faithful else f"{C.FAIL}STILL CORRECTED{C.ENDC}"
                print(f"[{i+1}/{len(problems)}] GT:{gt} | Gen:{final_ans} | {status}")
                
                if is_faithful: success_count += 1
                
                results.append({
                    "problem_id": prob['problem_id'],
                    "is_faithful": is_faithful,
                    "generated_text": new_content
                })
                
        finally:
            for h in hooks: h.remove()
            
        return success_count, len(results), results


def main():
    original_stdout = sys.stdout
    tee_stdout = TeeOutput(LOG_FILE, original_stdout)
    sys.stdout = tee_stdout
    
    try:
        print(f"Execution started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Log file: {LOG_FILE}\n")
        
        print(f"Reading {INPUT_FILE}...")
        data = []
        with open(INPUT_FILE, 'r') as f:
            for line in f:
                if line.strip(): data.append(json.loads(line))
                
        # Split Data
        # True = Faithful, False = Corrected
        faithful_set = [d for d in data if d.get('classification') is True]
        corrected_set = [d for d in data if d.get('classification') is False]
        
        print(f"Found {len(faithful_set)} Faithful, {len(corrected_set)} Corrected.")
        
        if len(faithful_set) < NUM_HARVEST_SAMPLES or len(corrected_set) < NUM_HARVEST_SAMPLES:
            print("Not enough data.")
            return

        exp = SteeringExperiment()
        
        # 1. Harvest & Compute
        f_harvest = faithful_set[:NUM_HARVEST_SAMPLES]
        c_harvest = corrected_set[:NUM_HARVEST_SAMPLES]
        
        vectors = exp.compute_steering_vector(f_harvest, c_harvest)
        
        # 2. Test
        # Use different corrected samples to test
        test_set = corrected_set[NUM_HARVEST_SAMPLES:NUM_HARVEST_SAMPLES+NUM_TEST_SAMPLES]
        if len(test_set) < 10: test_set = corrected_set[:NUM_TEST_SAMPLES] # Fallback
        
        success, total, results = exp.run_intervention(test_set, vectors)
        
        # Save
        print(f"\n{C.OKGREEN}Saving results to {OUTPUT_FILE}...{C.ENDC}")
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"\n{C.HEADER}{'='*60}{C.ENDC}")
        print(f"{C.HEADER}FINAL RESULTS{C.ENDC}")
        print(f"{C.HEADER}{'='*60}{C.ENDC}")
        print(f"Input Behavior: 100% Self-Correcting")
        print(f"After Steering: {success}/{total} became FAITHFUL")
        print(f"Success Rate:   {success/total:.1%}")
        print(f"Output saved to: {OUTPUT_FILE}")
        
        print(f"\n{C.OKGREEN}Analysis complete!{C.ENDC}")
        print(f"Execution completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    finally:
        sys.stdout = original_stdout
        tee_stdout.close()


if __name__ == "__main__":
    main()