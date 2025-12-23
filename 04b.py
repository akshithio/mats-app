import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

# --- CONFIGURATION ---
INPUT_FILE = "01d.jsonl"
MODEL_NAME = "Qwen/Qwen2.5-Math-7B-Instruct" 
NOISE_STD = 8.0   # Bumped up slightly to force separation
TRIALS_PER_SAMPLE = 5  # Run 5 noise seeds per problem to smooth variance
LAYERS_TO_SCAN = [0, 5, 10, 15, 20, 25] 
GENERATION_STEPS = 15

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

        if hidden.dim() == 2: hidden = hidden.unsqueeze(0)
        seq_len = hidden.shape[1]
        
        if self.mode == "noisy":
            if self.current_step <= self.injection_pos < self.current_step + seq_len:
                local_idx = self.injection_pos - self.current_step
                noise = torch.randn_like(hidden[:, local_idx, :]) * self.noise_std
                hidden[:, local_idx, :] += noise
                
        if self.current_step + seq_len > self.injection_pos:
            start_idx = max(0, self.injection_pos - self.current_step)
            relevant_data = hidden[:, start_idx:, :].detach().cpu()
            if self.mode == "clean": self.clean_activations.append(relevant_data)
            else: self.noisy_activations.append(relevant_data)
        
        self.current_step += seq_len
        if is_tuple: return (hidden,) + output[1:]
        else: return hidden

    def __enter__(self):
        target_layer = self.model.model.layers[self.layer_idx]
        self.hook_handle = target_layer.register_forward_hook(self.hook_fn)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.hook_handle: self.hook_handle.remove()

def load_data(filename):
    data = []
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            for line in f:
                try: data.append(json.loads(line))
                except: continue
    return data

def cosine_similarity(a, b):
    a_norm = F.normalize(a, p=2, dim=-1)
    b_norm = F.normalize(b, p=2, dim=-1)
    return (a_norm * b_norm).sum(dim=-1).numpy()

def main():
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto", torch_dtype=torch.float16)
    
    raw_data = load_data(INPUT_FILE)
    dataset = [d for d in raw_data if d.get('is_faithful') is not None]
    
    print(f"Scanning {len(LAYERS_TO_SCAN)} layers. Data size: {len(dataset)}. Trials per sample: {TRIALS_PER_SAMPLE}")
    
    # Store mean similarity per layer
    layer_results = {l: {"faithful": [], "stubborn": []} for l in LAYERS_TO_SCAN}
    
    for layer_idx in tqdm(LAYERS_TO_SCAN, desc="Layer Scan"):
        for item in dataset:
            try:
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
            except: continue

            hook_mgr = ResidualStreamHook(model, layer_idx, injection_idx, noise_std=NOISE_STD)
            
            # 1. Clean Run (Once)
            hook_mgr.mode = "clean"; hook_mgr.current_step = 0; hook_mgr.clean_activations = []
            with hook_mgr:
                with torch.no_grad(): model.generate(input_ids, max_new_tokens=GENERATION_STEPS, do_sample=False, pad_token_id=tokenizer.eos_token_id)
            
            if not hook_mgr.clean_activations: continue
            clean_traj = torch.cat(hook_mgr.clean_activations, dim=1).squeeze(0)

            # 2. Noisy Runs (Multiple Trials)
            sample_sims = []
            for _ in range(TRIALS_PER_SAMPLE):
                hook_mgr.mode = "noisy"; hook_mgr.current_step = 0; hook_mgr.noisy_activations = []
                with hook_mgr:
                    with torch.no_grad(): model.generate(input_ids, max_new_tokens=GENERATION_STEPS, do_sample=False, pad_token_id=tokenizer.eos_token_id)
                
                if hook_mgr.noisy_activations:
                    noisy_traj = torch.cat(hook_mgr.noisy_activations, dim=1).squeeze(0)
                    min_len = min(clean_traj.shape[0], noisy_traj.shape[0])
                    if min_len >= 5:
                        sim = cosine_similarity(clean_traj[:min_len], noisy_traj[:min_len]).mean()
                        sample_sims.append(sim)
            
            # Average the trials for this sample
            if sample_sims:
                avg_sim = np.mean(sample_sims)
                if item['is_faithful']:
                    layer_results[layer_idx]["faithful"].append(avg_sim)
                else:
                    layer_results[layer_idx]["stubborn"].append(avg_sim)

    layers = sorted(LAYERS_TO_SCAN)
    f_means = [np.mean(layer_results[l]["faithful"]) if layer_results[l]["faithful"] else 0 for l in layers]
    s_means = [np.mean(layer_results[l]["stubborn"]) if layer_results[l]["stubborn"] else 0 for l in layers]
    
    f_err = [np.std(layer_results[l]["faithful"])/np.sqrt(len(layer_results[l]["faithful"])) if layer_results[l]["faithful"] else 0 for l in layers]
    s_err = [np.std(layer_results[l]["stubborn"])/np.sqrt(len(layer_results[l]["stubborn"])) if layer_results[l]["stubborn"] else 0 for l in layers]

    plt.figure(figsize=(10, 6))
    plt.errorbar(layers, f_means, yerr=f_err, fmt='r--o', label="Faithful (Group A)", linewidth=2, capsize=5)
    plt.errorbar(layers, s_means, yerr=s_err, fmt='b-o', label="Stubborn (Group B)", linewidth=3, capsize=5)
    
    plt.xlabel("Layer Depth (Input -> Output)")
    plt.ylabel("Mean Cosine Similarity (Stiffness)")
    plt.title(f"Stiffness Tomography (Averaged over {TRIALS_PER_SAMPLE} noise seeds)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("tomography_plot.png")
    print("Saved tomography_plot.png")

if __name__ == "__main__":
    main()