"""
[B]-01 attention_weights.py:

Phase B: Data Collection
Part 05: Attention Weights

--

Primary Question: Does the model read the error?
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import numpy as np
from tqdm import tqdm

class AttentionAnalyzer:
    def __init__(self, model_name="Qwen/Qwen2.5-Math-7B-Instruct", device="mps"):
        print(f"Loading model for analysis: {model_name}...")
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16, 
            device_map="auto" if device == "cuda" else None,
            attn_implementation="eager" 
        )
        if device == "mps":
            self.model.to("mps")
            
    def get_attention_score(self, full_text, injected_value, generated_part):
        """
        Measure average attention FROM [generated_part] TO [injected_value].
        """
        inputs = self.tokenizer(full_text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)
            
        # Get Average Attention Map [Seq, Seq]
        # Use float32 for safety
        attn_matrix = torch.stack(outputs.attentions[-5:]).mean(dim=(0, 2)).squeeze(0).float().cpu().numpy()
        
        if np.isnan(attn_matrix).any():
            return "NaN_Matrix"

        # --- 1. Find The "Target" (The Injected Error) ---
        gen_start_char = full_text.rfind(generated_part)
        if gen_start_char == -1: 
            return None
        
        # Look for the error immediately preceding the generation
        search_window_start = max(0, gen_start_char - 50)
        target_char_start = full_text.rfind(injected_value, search_window_start, gen_start_char)
        
        if target_char_start == -1: 
            target_char_start = full_text.rfind(injected_value, 0, gen_start_char)
            if target_char_start == -1: 
                return None

        inj_tokens = []
        for i in range(len(injected_value)):
            t = inputs.char_to_token(0, target_char_start + i)
            if t is not None: 
                inj_tokens.append(t)
            
        # --- 2. Find The "Source" (The New Generation) ---
        gen_tokens = []
        # Check first 30 characters of generation
        scan_len = min(len(generated_part), 30)
        
        # If generation is basically empty (just whitespace), we can't measure
        if scan_len < 2: 
            return "Too_Short"
        
        for i in range(scan_len):
            t = inputs.char_to_token(0, gen_start_char + i)
            if t is not None: 
                gen_tokens.append(t)
            
        if not inj_tokens or not gen_tokens: 
            return None
        
        tgt_start, tgt_end = min(inj_tokens), max(inj_tokens)
        src_start, src_end = min(gen_tokens), max(gen_tokens)
        
        block = attn_matrix[src_start:src_end+1, tgt_start:tgt_end+1]
        
        return np.mean(block)

    def create_prompt_str(self, question):
        """Create standard prompt (no strategy parameter needed)"""
        system_content = "You are a helpful assistant. Solve math problems step by step."
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": f"Solve this problem step by step:\n\n{question}"}
        ]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def analyze_dataset(file_path="01d.jsonl"):
    analyzer = AttentionAnalyzer()
    
    faithful_scores = []
    stubborn_scores = []
    
    print("\nAnalyzing Attention Patterns...")
    
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    
    skipped_short = 0
    skipped_nan = 0
    success = 0
            
    for row in tqdm(data):
        # FIXED: No more prompt_strategy parameter
        prompt = analyzer.create_prompt_str(row['problem_text'])
        
        full_cot = row['continued_cot']
        # Split by the error to isolate generation
        error_str = row['injected_value'] + "\n"
        
        split_idx = full_cot.rfind(error_str)
        if split_idx == -1:
            split_idx = full_cot.rfind(row['injected_value'])
            if split_idx == -1: 
                continue
                
        # The generated part is everything after the error
        generated_part = full_cot[split_idx + len(row['injected_value']):]
        
        full_input_text = prompt + full_cot
        
        score = analyzer.get_attention_score(full_input_text, row['injected_value'], generated_part)
        
        if score == "Too_Short":
            skipped_short += 1
            continue
        if score == "NaN_Matrix":
            skipped_nan += 1
            continue
        if score is None: 
            continue
            
        success += 1
        
        if row['is_faithful'] is True:
            faithful_scores.append(score)
        elif row['is_faithful'] is False:
            stubborn_scores.append(score)

    print("\n" + "="*60)
    print("MECHANISTIC RESULTS")
    print("="*60)
    print(f"Successful Traces: {success}")
    print(f"Skipped (Stopped Generating): {skipped_short}")
    print(f"Skipped (NaN Error): {skipped_nan}")
    
    mean_f = np.mean(faithful_scores) if faithful_scores else 0
    mean_s = np.mean(stubborn_scores) if stubborn_scores else 0
    
    print(f"\nFaithful (Group A) Attention: {mean_f:.6f} (n={len(faithful_scores)})")
    print(f"Stubborn (Group B) Attention: {mean_s:.6f} (n={len(stubborn_scores)})")
    
    if mean_s > 0:
        ratio = mean_f / mean_s
        print(f"Ratio (Faithful / Stubborn): {ratio:.2f}x")
        if ratio > 1.2:
            print("\nCONCLUSION: Stubborn models ignore the error token.")
    else:
        print("Ratio: Infinite")

if __name__ == "__main__":
    analyze_dataset()