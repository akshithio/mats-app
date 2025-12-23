import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

INPUT_FILE = "01d.jsonl"
OUTPUT_FILE = "04a.jsonl"
MODEL_NAME = "Qwen/Qwen2.5-Math-7B-Instruct" 
NOISE_STD = 5.0
GENERATION_STEPS = 32
LAYER_FRAC = 0.5  

class ResidualStreamHook:
    def __init__(self, model, layer_idx, injection_pos, noise_std=0.0):
        self.model = model
        self.layer_idx = layer_idx
        self.injection_pos = injection_pos
        self.noise_std = noise_std
        self.clean_activations = []
        self.noisy_activations = []
        self.hook_handle = None
        self.mode = "clean"
        self.current_step = 0

    def hook_fn(self, module, args, output):
        if isinstance(output, tuple):
            hidden = output[0]
            is_tuple = True
        else:
            hidden = output
            is_tuple = False

        # --- FIX FOR 2D TENSOR ERROR (MPS/Batch=1) ---
        # If shape is [seq_len, hidden_dim], we view it as [1, seq_len, hidden_dim]
        # This ensures our slicing logic works.
        if hidden.dim() == 2:
            working_hidden = hidden.unsqueeze(0)
        else:
            working_hidden = hidden
        
        # working_hidden shape: [batch, seq_len, hidden_dim]
        seq_len = working_hidden.shape[1]
        
        # 1. NOISE INJECTION (Only in 'noisy' mode, at specific position)
        if self.mode == "noisy":
            # Check if the injection point falls within this chunk of tokens
            if self.current_step <= self.injection_pos < self.current_step + seq_len:
                local_idx = self.injection_pos - self.current_step
                
                # Create noise vector
                noise = torch.randn_like(working_hidden[:, local_idx, :]) * self.noise_std
                
                # Inject! (In-place modification of the view updates the original tensor)
                working_hidden[:, local_idx, :] += noise
                
        # 2. CACHING (Capture generated tokens AFTER injection)
        if self.current_step + seq_len > self.injection_pos:
            start_idx = max(0, self.injection_pos - self.current_step)
            
            # Detach to save memory, move to CPU
            relevant_data = working_hidden[:, start_idx:, :].detach().cpu()
            
            if self.mode == "clean":
                self.clean_activations.append(relevant_data)
            else:
                self.noisy_activations.append(relevant_data)
        
        self.current_step += seq_len
        
        # --- RETURN CORRECT TYPE ---
        # We modified 'working_hidden' in place. 
        # Since 'working_hidden' is a view of 'hidden' (or 'hidden' itself), 'hidden' is now noisy.
        if is_tuple:
            # Reconstruct tuple with modified tensor
            return (hidden,) + output[1:]
        else:
            # Return modified tensor directly
            return hidden

    def __enter__(self):
        target_layer = self.model.model.layers[self.layer_idx]
        self.hook_handle = target_layer.register_forward_hook(self.hook_fn)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.hook_handle:
            self.hook_handle.remove()

def load_data(filename):
    data = []
    if not os.path.exists(filename):
        print(f"Error: {filename} not found. Run 01d.py first.")
        return []
        
    with open(filename, 'r') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return data

def cosine_similarity(a, b):
    # a, b shape: [seq_len, hidden_dim]
    a_norm = F.normalize(a, p=2, dim=-1)
    b_norm = F.normalize(b, p=2, dim=-1)
    return (a_norm * b_norm).sum(dim=-1).numpy()

def plot_results(results):
    faithful_sims = []
    stubborn_sims = []
    
    for r in results:
        sims = r['similarities']
        target_len = 20
        if len(sims) >= target_len:
            sims = sims[:target_len]
            if r['is_faithful']:
                faithful_sims.append(sims)
            else:
                stubborn_sims.append(sims)
    
    if not faithful_sims or not stubborn_sims:
        print(f"Collecting data... (Faithful: {len(faithful_sims)}, Stubborn: {len(stubborn_sims)})")
        return

    f_mean = np.mean(faithful_sims, axis=0)
    s_mean = np.mean(stubborn_sims, axis=0)
    
    plt.figure(figsize=(10, 6))
    plt.plot(f_mean, label=f'Faithful Traces (Group A) n={len(faithful_sims)}', linestyle='--', color='red')
    plt.plot(s_mean, label=f'Stubborn Traces (Group B) n={len(stubborn_sims)}', linewidth=3, color='blue')
    
    plt.title(f"Reasoning Stiffness: Resilience to Residual Stream Noise (std={NOISE_STD})")
    plt.xlabel("Tokens Generated After Noise Injection")
    plt.ylabel("Cosine Similarity (Clean vs Noisy)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_img = "04a.png"
    plt.savefig(output_img)
    print(f"\nPlot updated: {output_img}")

def main():
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        device_map="auto", 
        torch_dtype=torch.float16
    )
    
    target_layer_idx = int(len(model.model.layers) * LAYER_FRAC)
    print(f"Injecting noise at Layer {target_layer_idx}")
    
    raw_data = load_data(INPUT_FILE)
    dataset = [d for d in raw_data if d.get('is_faithful') is not None]
    print(f"Found {len(dataset)} usable samples.")
    
    results = []
    
    for item in tqdm(dataset, desc="Stress Testing"):
        prompt = f"<|im_start|>system\nYou are a helpful assistant. Solve math problems step by step.<|im_end|>\n<|im_start|>user\nSolve this problem step by step:\n\n{item['problem_text']}<|im_end|>\n<|im_start|>assistant\n"
        full_text_original = item['original_cot']
        
        try:
            # Reconstruct injection point
            context = item['injection_point']
            val = item['original_value']
            
            c_start = full_text_original.find(context)
            if c_start == -1: continue
            
            v_start = context.find(val)
            if v_start == -1: continue
            
            cut_char_index = c_start + v_start + len(val) 
            prefix_text = prompt + full_text_original[:cut_char_index]
            
            input_ids = tokenizer(prefix_text, return_tensors="pt").input_ids.to(model.device)
            injection_idx = input_ids.shape[1] - 1 
            
        except Exception:
            continue

        hook_mgr = ResidualStreamHook(model, target_layer_idx, injection_idx, noise_std=NOISE_STD)
        
        # 1. Clean Run
        hook_mgr.mode = "clean"
        hook_mgr.current_step = 0
        hook_mgr.clean_activations = []
        with hook_mgr:
            with torch.no_grad():
                model.generate(input_ids, max_new_tokens=GENERATION_STEPS, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        
        # 2. Noisy Run
        hook_mgr.mode = "noisy"
        hook_mgr.current_step = 0
        hook_mgr.noisy_activations = []
        with hook_mgr:
            with torch.no_grad():
                model.generate(input_ids, max_new_tokens=GENERATION_STEPS, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        
        # 3. Compute Stats
        if not hook_mgr.clean_activations or not hook_mgr.noisy_activations:
            continue
            
        try:
            clean_traj = torch.cat(hook_mgr.clean_activations, dim=1).squeeze(0)
            noisy_traj = torch.cat(hook_mgr.noisy_activations, dim=1).squeeze(0)
        except Exception:
            continue
            
        min_len = min(clean_traj.shape[0], noisy_traj.shape[0])
        if min_len < 2: continue
        
        clean_traj = clean_traj[:min_len]
        noisy_traj = noisy_traj[:min_len]
        
        sims = cosine_similarity(clean_traj, noisy_traj)
        
        record = {
            "problem_id": item['problem_id'],
            "is_faithful": item['is_faithful'], 
            "similarities": sims.tolist()
        }
        results.append(record)
        
        with open(OUTPUT_FILE, "a") as f:
            f.write(json.dumps(record) + "\n")

    plot_results(results)

if __name__ == "__main__":
    main()