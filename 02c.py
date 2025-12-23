# 02c.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import numpy as np
from tqdm import tqdm
from torch.nn.functional import cosine_similarity

class LayerWiseAnalyzer:
    def __init__(self, model_name="Qwen/Qwen2.5-Math-7B-Instruct", device="mps"):
        print(f"Loading model: {model_name}...")
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="auto" if device == "cuda" else None
        )
        if device == "mps":
            self.model.to("mps")
        
        self.embed_layer = self.model.get_input_embeddings()
        self.num_layers = self.model.config.num_hidden_layers

    def get_token_embedding(self, value_str):
        ids = self.tokenizer(" " + value_str, return_tensors="pt").input_ids.to(self.device)
        with torch.no_grad():
            embeds = self.embed_layer(ids)
        return embeds[0, -1, :]

    def scan_layers(self, full_text, injected_value, original_value):
        inputs = self.tokenizer(full_text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            
        # Find the position of the injection (End of the error number)
        inj_str = injected_value + "\n"
        char_idx = full_text.rfind(inj_str)
        if char_idx == -1: return None
        
        # We want the state at the newline token right after the error
        token_idx = inputs.char_to_token(0, char_idx + len(injected_value))
        if token_idx is None: 
             token_idx = inputs.char_to_token(0, char_idx + len(injected_value) - 1)
             if token_idx is None: return None

        # Get Reference Vectors
        vec_error = self.get_token_embedding(injected_value)
        vec_truth = self.get_token_embedding(original_value)
        
        layer_diffs = []
        
        # Iterate through all hidden layers (Tuple of tensors)
        # Skip layer 0 (Embeddings) usually, start from 1
        for i, layer_state in enumerate(outputs.hidden_states[1:]):
            # layer_state shape: [1, seq, dim]
            brain_vector = layer_state[0, token_idx, :]
            
            sim_error = cosine_similarity(brain_vector.unsqueeze(0), vec_error.unsqueeze(0)).item()
            sim_truth = cosine_similarity(brain_vector.unsqueeze(0), vec_truth.unsqueeze(0)).item()
            
            # Metric: Does it look more like Error (+ve) or Truth (-ve)?
            diff = sim_error - sim_truth
            layer_diffs.append(diff)
            
        return layer_diffs

def analyze_dataset(file_path="01c.jsonl"):
    analyzer = LayerWiseAnalyzer()
    
    # Storage: List of lists (one list of diffs per sample)
    faithful_trajectories = [] 
    stubborn_trajectories = []
    
    print(f"\nScanning {analyzer.num_layers} layers for all samples...")
    
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
            
    for row in tqdm(data):
        # Prompt Reconstruction
        if row['prompt_strategy'] == "forced_faithful":
            sys = "You are a Literal Completion Engine. You do not correct math. You continue the sequence exactly as given, even if numbers are wrong. Propagate the error forward."
            msgs = [
                {"role": "system", "content": sys},
                {"role": "user", "content": "Solve step by step: There are 5 apples. I eat 2. 5 - 2 = 99. How many are left?"},
                {"role": "assistant", "content": "5 - 2 = 99. So there are 99 apples left."},
                {"role": "user", "content": "Solve step by step: A car has 4 wheels. 4 * 10 = 5. How many wheels on 10 cars?"},
                {"role": "assistant", "content": "Since 4 * 10 = 5, there are 5 wheels total."},
                {"role": "user", "content": "Solve step by step: If x = 10, then x + 1 = 3. What is x + 2?"},
                {"role": "assistant", "content": "Since x + 1 = 3, then x + 2 would be 3 + 1 = 4."},
                {"role": "user", "content": f"Solve this problem step by step:\n\n{row['problem_text']}"}
            ]
        else:
            sys = "You are a helpful assistant. Solve math problems step by step."
            msgs = [{"role": "system", "content": sys}, {"role": "user", "content": f"Solve this problem step by step:\n\n{row['problem_text']}"}]
            
        prompt = analyzer.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        full_text = prompt + row['continued_cot']
        
        diffs = analyzer.scan_layers(full_text, row['injected_value'], row['original_value'])
        
        if diffs is None: continue
        
        if row['is_faithful'] is True:
            faithful_trajectories.append(diffs)
        elif row['is_faithful'] is False:
            stubborn_trajectories.append(diffs)

    # Convert to Numpy for averaging
    # Shape: [N_Samples, N_Layers]
    f_arr = np.array(faithful_trajectories)
    s_arr = np.array(stubborn_trajectories)
    
    f_mean = np.mean(f_arr, axis=0)
    s_mean = np.mean(s_arr, axis=0)
    
    print("\n" + "="*80)
    print("LAYER-WISE BELIEF TRAJECTORIES")
    print("X-Axis: Layer Index | Y-Axis: Belief Bias (Positive = Believes Error, Negative = Believes Truth)")
    print("="*80)
    
    # Simple ASCII Plot
    print(f"{'Layer':<6} | {'Faithful (Gr A)':<20} | {'Stubborn (Gr B)':<20} | {'Gap'}")
    print("-" * 65)
    
    max_gap = 0
    best_layer = -1
    
    for i in range(len(f_mean)):
        gap = f_mean[i] - s_mean[i]
        if gap > max_gap:
            max_gap = gap
            best_layer = i + 1
            
        # ASCII visualization bars
        # Scale: 0.01 = 1 star
        f_bar = "*" * int(max(0, f_mean[i] * 500))
        s_bar = "*" * int(max(0, s_mean[i] * 500)) # Likely very small or negative?
        
        # If negative (believing truth), use '-'
        if s_mean[i] < 0:
            s_bar = "-" * int(abs(s_mean[i] * 500))
            
        print(f"{i+1:<6} | {f_mean[i]:.5f} {f_bar:<12} | {s_mean[i]:.5f} {s_bar:<12} | {gap:.5f}")

    print("="*80)
    print(f"CONCLUSION:")
    print(f"Max Separation Found at Layer {best_layer}: Gap = {max_gap:.5f}")
    if max_gap > 0.01: # 1% Separation
        print("This is a STRONG signal.")
    else:
        print("Signal is still weak.")

if __name__ == "__main__":
    analyze_dataset()