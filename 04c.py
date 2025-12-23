import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import re

# --- CONFIGURATION ---
INPUT_FILE = "01d.jsonl"
MODEL_NAME = "Qwen/Qwen2.5-Math-7B-Instruct" 
LAYER_IDX = 5  # The "Sweet Spot" you found in Experiment 2
NOISE_LEVELS = [0, 4, 8, 12, 16, 20, 24] # The Sweep
TRIALS = 3     # Run 3 times per noise level to average
MAX_NEW_TOKENS = 64

class NoiseInjectionHook:
    def __init__(self, model, layer_idx, injection_pos, noise_std=0.0):
        self.model = model
        self.layer_idx = layer_idx
        self.injection_pos = injection_pos
        self.noise_std = noise_std
        self.hook_handle = None
        self.active = False
        self.current_step = 0

    def hook_fn(self, module, args, output):
        if not self.active: return output
        
        if isinstance(output, tuple):
            hidden = output[0]
            is_tuple = True
        else:
            hidden = output
            is_tuple = False
            
        if hidden.dim() == 2: hidden = hidden.unsqueeze(0)
        seq_len = hidden.shape[1]
        
        # Inject only at the specific token position
        if self.current_step <= self.injection_pos < self.current_step + seq_len:
            local_idx = self.injection_pos - self.current_step
            noise = torch.randn_like(hidden[:, local_idx, :]) * self.noise_std
            hidden[:, local_idx, :] += noise
        
        self.current_step += seq_len
        if is_tuple: return (hidden,) + output[1:]
        else: return hidden

    def __enter__(self):
        target_layer = self.model.model.layers[self.layer_idx]
        self.hook_handle = target_layer.register_forward_hook(self.hook_fn)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.hook_handle: self.hook_handle.remove()

def extract_answer(text):
     # 1. Look for the standard GSM8K separator "####"
    if "####" in text:
        after_hash = text.split("####")[-1]
        # Remove commas (e.g., "1,234" -> "1234")
        clean_text = after_hash.replace(",", "")
        matches = re.findall(r'-?\d+\.?\d*', clean_text)
        if matches: return float(matches[0])
    
    # 2. Look for "The answer is X" pattern
    answer_pattern = re.search(r'(?:answer is|result is|=)\s*(-?[\d,]+\.?\d*)', text, re.IGNORECASE)
    if answer_pattern:
        clean_num = answer_pattern.group(1).replace(",", "")
        return float(clean_num)

    # 3. Fallback: Last number, but strip commas first
    clean_text = text.replace(",", "")
    matches = re.findall(r'-?\d+\.?\d*', clean_text)
    if not matches: return None
    return float(matches[-1])

def load_data(filename):
    data = []
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            for line in f:
                try: data.append(json.loads(line))
                except: continue
    return data

def main():
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto", torch_dtype=torch.float16)
    
    raw_data = load_data(INPUT_FILE)
    dataset = [d for d in raw_data if d.get('is_faithful') is not None]
    # Limit dataset for speed if needed
    dataset = dataset[:50] 
    
    results = {lvl: {"faithful_survival": [], "stubborn_survival": []} for lvl in NOISE_LEVELS}
    
    print(f"Running Basin Capacity Sweep on Layer {LAYER_IDX}...")
    
    for item in tqdm(dataset, desc="Samples"):
        try:
            # Reconstruct Prompt
            full_text_original = item['original_cot']
            context = item['injection_point']
            val = item['original_value']
            c_start = full_text_original.find(context)
            if c_start == -1: continue
            v_start = context.find(val)
            if v_start == -1: continue
            cut_char_index = c_start + v_start + len(val) 
            
            prompt = f"<|im_start|>system\nYou are a helpful assistant. Solve math problems step by step.<|im_end|>\n<|im_start|>user\nSolve this problem step by step:\n\n{item['problem_text']}<|im_end|>\n<|im_start|>assistant\n"
            prefix_text = prompt + full_text_original[:cut_char_index]
            input_ids = tokenizer(prefix_text, return_tensors="pt").input_ids.to(model.device)
            injection_idx = input_ids.shape[1] - 1
            
            # Get the "Target" answer (what the model produced originally)
            original_output = item['final_answer']
            
        except: continue

        for noise in NOISE_LEVELS:
            survived_count = 0
            for _ in range(TRIALS):
                hook = NoiseInjectionHook(model, LAYER_IDX, injection_idx, noise_std=noise)
                hook.active = True
                hook.current_step = 0
                
                with hook:
                    with torch.no_grad():
                        out = model.generate(input_ids, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, pad_token_id=tokenizer.eos_token_id)
                
                gen_text = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
                new_ans = extract_answer(gen_text)
                
                if new_ans is not None and abs(new_ans - original_output) < 0.1:
                    survived_count += 1
            
            survival_rate = survived_count / TRIALS
            if item['is_faithful']:
                results[noise]["faithful_survival"].append(survival_rate)
            else:
                results[noise]["stubborn_survival"].append(survival_rate)

    # Plotting
    f_means = [np.mean(results[n]["faithful_survival"]) if results[n]["faithful_survival"] else 0 for n in NOISE_LEVELS]
    s_means = [np.mean(results[n]["stubborn_survival"]) if results[n]["stubborn_survival"] else 0 for n in NOISE_LEVELS]
    
    plt.figure(figsize=(10, 6))
    plt.plot(NOISE_LEVELS, f_means, 'r--o', label="Faithful (Group A)", linewidth=2)
    plt.plot(NOISE_LEVELS, s_means, 'b-o', label="Stubborn (Group B)", linewidth=3)
    plt.xlabel(f"Noise Magnitude (std) injected at Layer {LAYER_IDX}")
    plt.ylabel("Probability of Retaining Original Answer")
    plt.title("Basin Capacity: Functional Robustness to Noise")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.1, 1.1)
    plt.savefig("basin_capacity_plot.png")
    print("Saved basin_capacity_plot.png")

if __name__ == "__main__":
    main()