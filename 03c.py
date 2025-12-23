import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import numpy as np
from tqdm import tqdm

class LogitLens:
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
            
        self.lm_head = self.model.lm_head
        self.num_layers = self.model.config.num_hidden_layers

    def get_token_rank(self, logits, target_str):
        target_ids = self.tokenizer(target_str, add_special_tokens=False).input_ids
        if not target_ids: return 150000 # Fallback
        
        target_id = target_ids[-1]
        
        # Get Rank
        # We don't need full sort (slow), just count how many are greater
        score = logits[target_id].item()
        # Count elements > score
        rank = (logits > score).sum().item()
        return rank

    def scan_all_layers(self, full_text, original_value):
        inputs = self.tokenizer(full_text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            
        # Find position: The Newline after the injected error
        # We want to see if the model immediately regrets the error
        char_idx = full_text.rfind("\n") 
        if char_idx == -1: return None
        
        token_idx = inputs.char_to_token(0, char_idx - 1) # Check state AT the number
        if token_idx is None: return None

        layer_ranks = []
        
        # Scan layers 1 to End
        for i, layer_state in enumerate(outputs.hidden_states[1:]):
            hidden_state = layer_state[0, token_idx, :]
            hidden_state = hidden_state.to(self.model.lm_head.weight.dtype)
            logits = self.model.lm_head(hidden_state)
            
            rank = self.get_token_rank(logits, original_value)
            layer_ranks.append(rank)
            
        return layer_ranks

def analyze_dataset(file_path="01c.jsonl"):
    lens = LogitLens()
    
    # Store: [Layer] -> List of Ranks
    faithful_layers = [[] for _ in range(lens.num_layers)]
    stubborn_layers = [[] for _ in range(lens.num_layers)]
    
    print(f"\nScanning {lens.num_layers} layers for Truth Resurfacing...")
    
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
            
    for row in tqdm(data):
        # Prompt reconstruction
        if row['prompt_strategy'] == "forced_faithful":
            sys = "You are a Literal Completion Engine. You do not correct math. You continue the sequence exactly as given, even if numbers are wrong. Propagate the error forward."
            msgs = [{"role": "system", "content": sys}, {"role": "user", "content": f"Solve this problem step by step:\n\n{row['problem_text']}"}]
        else:
            sys = "You are a helpful assistant. Solve math problems step by step."
            msgs = [{"role": "system", "content": sys}, {"role": "user", "content": f"Solve this problem step by step:\n\n{row['problem_text']}"}]
            
        prompt = lens.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        full_text = prompt + row['continued_cot']
        
        ranks = lens.scan_all_layers(full_text, row['original_value'])
        if ranks is None: continue
        
        for i, r in enumerate(ranks):
            if row['is_faithful']:
                faithful_layers[i].append(r)
            else:
                stubborn_layers[i].append(r)

    print("\n" + "="*80)
    print("LOGIT LENS: MEDIAN RANK OF 'TRUTH'")
    print("Lower # = Closer to speaking the truth")
    print("="*80)
    print(f"{'Layer':<6} | {'Faithful Rank':<15} | {'Stubborn Rank':<15} | {'Diff'}")
    print("-" * 65)
    
    for i in range(lens.num_layers):
        f_med = np.median(faithful_layers[i]) if faithful_layers[i] else 0
        s_med = np.median(stubborn_layers[i]) if stubborn_layers[i] else 0
        
        # Visualization
        diff = f_med - s_med
        indicator = "<<< STUBBORN WINS" if diff > 1000 else ""
        
        print(f"{i+1:<6} | {int(f_med):<15} | {int(s_med):<15} | {indicator}")

if __name__ == "__main__":
    analyze_dataset()