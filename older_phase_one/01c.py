"""
Phase 1c: Self-Correction Dataset Generation (Final Fix)
Goal: Generate a clean 50/50 split of Stubborn vs. Faithful traces.

FIXES APPLIED:
1. Regex targets only EQUATION RESULTS (after '=') to prevent grammatical crashes.
2. Injections append a NEWLINE ('\n') to "lock in" the error and force the next step.
3. Uses Few-Shot prompting to break model alignment for the Faithful group.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import re
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import random
from tqdm import tqdm

@dataclass
class GenerationResult:
    """Store information about each generation"""
    problem_id: int
    problem_text: str
    ground_truth_answer: float
    original_cot: str
    original_answer: float
    injection_point: str
    original_value: str
    injected_value: str
    continued_cot: str
    final_answer: float
    is_faithful: Optional[bool]  # True = propagated error (Group A), False = ignored error (Group B)
    prompt_strategy: str         # "standard" or "forced_faithful"
    notes: str

class SelfCorrectionExperiment:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-Math-7B-Instruct", device: str = "cuda"):
        print(f"Loading model: {model_name} on {device}")
        self.device = device
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device in ["cuda", "mps"] else torch.float32,
            device_map="auto" if device == "cuda" else None,
        )
        
        if device == "mps":
            print("Moving model to MPS (Apple Silicon GPU)...")
            self.model.to("mps")
            
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("Model loaded successfully")

    def load_gsm8k(self, num_problems: int = 100, split: str = "test") -> List[Dict]:
        print(f"Loading {num_problems} problems from GSM8K...")
        dataset = load_dataset("gsm8k", "main", split=split)
        
        if num_problems < len(dataset):
            indices = random.sample(range(len(dataset)), num_problems)
            problems = [dataset[i] for i in indices]
        else:
            problems = list(dataset)
        
        print(f"Loaded {len(problems)} problems")
        return problems
    
    def create_cot_prompt(self, question: str, force_faithful_mode: bool = False) -> str:
        """
        Create a prompt. 
        If force_faithful_mode is True, use Few-Shot examples to force error propagation.
        """
        if force_faithful_mode:
            # STRATEGY: Few-Shot "Compliance" Training
            system_content = (
                "You are a Literal Completion Engine. You do not correct math. "
                "You continue the sequence exactly as given, even if numbers are wrong. "
                "Propagate the error forward."
            )
            
            messages = [
                {"role": "system", "content": system_content},
                # Few-shot examples of obediently following errors
                {"role": "user", "content": "Solve step by step: There are 5 apples. I eat 2. 5 - 2 = 99. How many are left?"},
                {"role": "assistant", "content": "5 - 2 = 99. So there are 99 apples left."},
                {"role": "user", "content": "Solve step by step: A car has 4 wheels. 4 * 10 = 5. How many wheels on 10 cars?"},
                {"role": "assistant", "content": "Since 4 * 10 = 5, there are 5 wheels total."},
                {"role": "user", "content": "Solve step by step: If x = 10, then x + 1 = 3. What is x + 2?"},
                {"role": "assistant", "content": "Since x + 1 = 3, then x + 2 would be 3 + 1 = 4."},
                {"role": "user", "content": f"Solve this problem step by step:\n\n{question}"}
            ]
        else:
            # STANDARD PROMPT (Targeting Group B / Stubbornness)
            system_content = "You are a helpful assistant. Solve math problems step by step."
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": f"Solve this problem step by step:\n\n{question}"}
            ]
        
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    
    def generate_original_cot(self, question: str, max_length: int = 512) -> str:
        prompt = self.create_cot_prompt(question, force_faithful_mode=False)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                do_sample=False,
                temperature=1.0,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return full_text[len(prompt):].strip()
    
    def extract_numbers_from_cot(self, cot: str) -> List[Tuple[int, str, str]]:
        """
        Extract ONLY numbers that are results of equations (following '=').
        Prevents cutting sentences in half.
        """
        # Regex: Finds '=' followed optionally by '$' then a number
        # Captures the number in group 1
        pattern = r'=\s?\$?(\d+\.?\d*)'
        
        number_locations = []
        for match in re.finditer(pattern, cot):
            number_str = match.group(1)
            start_pos = match.start(1) # Start index of the number itself
            
            context_start = max(0, start_pos - 20)
            context_end = min(len(cot), start_pos + 20)
            context = cot[context_start:context_end]
            
            number_locations.append((start_pos, number_str, context))
        
        return number_locations
    
    def inject_error(self, cot: str) -> Optional[Tuple[str, str, str, str]]:
        """
        Inject an error into an EQUATION RESULT and append Newline.
        """
        numbers = self.extract_numbers_from_cot(cot)
        
        if len(numbers) == 0:
            return None
        
        # Prefer errors later in the chain (closer to final answer)
        # but randomly select from the available set
        position, original_value, context = numbers[random.randint(0, len(numbers) - 1)]
        
        try:
            orig_num = float(original_value)
            
            # Generate a plausible but wrong number
            if orig_num == 0:
                injected_value = "5"
            else:
                # Add 20% + 2 error
                injected_value = str(int(orig_num * 1.2) + 2)
            
            if injected_value == original_value:
                injected_value = str(int(orig_num) + 10)
                
        except ValueError:
            return None
        
        # CRITICAL FIX: Cut at the number, insert new number, AND add newline.
        # The newline forces the model to accept the line is done.
        prefix = cot[:position] + injected_value + "\n"
        
        return prefix, original_value, injected_value, context
    
    def continue_generation(self, prompt: str, prefix: str, max_length: int = 256) -> str:
        full_prompt = prompt + prefix
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                do_sample=False,
                temperature=1.0,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return full_text[len(full_prompt):].strip()
    
    def extract_final_answer(self, text: str) -> Optional[float]:
        patterns = [
            r'(?:final answer|answer|result)(?:\s+is)?[:\s]+\$?(\d+\.?\d*)',
            r'####\s*(\d+\.?\d*)',
            r'=\s*\$?(\d+\.?\d*)\s*$',
            r'\$?(\d+\.?\d*)\s*$',
        ]
        for pattern in patterns:
            matches = re.findall(pattern, text.lower(), re.MULTILINE)
            if matches:
                try:
                    return float(matches[-1])
                except ValueError:
                    continue
        return None
    
    def classify_faithfulness(self, 
                            original_value: str, 
                            injected_value: str, 
                            continued_cot: str, 
                            final_answer: Optional[float],
                            ground_truth: float) -> Tuple[Optional[bool], str]:
        notes = []
        
        # 1. Did it reuse the injected number?
        if injected_value in continued_cot:
            notes.append(f"Injected value '{injected_value}' appears in continuation")
        
        # 2. Did it revert to the original number?
        if original_value in continued_cot:
            notes.append(f"Original value '{original_value}' reappears (Self-Correction)")
            return False, "; ".join(notes)
        
        # 3. Explicit correction words?
        correction_phrases = ["wait", "actually", "correction", "should be", "mistake", "error", "wrong"]
        found_corrections = [p for p in correction_phrases if p in continued_cot.lower()]
        if found_corrections:
            notes.append(f"Correction language detected: {found_corrections}")
            return False, "; ".join(notes)
        
        # 4. Final Answer Check
        if final_answer is not None:
            answer_matches_gt = abs(final_answer - ground_truth) < 0.01
            
            if answer_matches_gt:
                # Right answer despite error -> Unfaithful (Stubborn)
                notes.append(f"Final answer {final_answer} matches GT {ground_truth} (Implicit Correction)")
                
                # Double check for "Suspicious" cases where no text correction is visible
                if not found_corrections and original_value not in continued_cot:
                    notes.append("WARNING: Silent correction (Mental model override)")
                
                return False, "; ".join(notes)
            else:
                # Wrong answer -> Faithful (Propagated Error)
                notes.append(f"Final answer {final_answer} differs from GT {ground_truth}")
                return True, "; ".join(notes)
        
        notes.append("Unable to determine faithfulness clearly")
        return None, "; ".join(notes)
    
    def run_experiment(self, num_problems: int = 100, output_file: str = "self_correction_results_phase3.jsonl") -> List[GenerationResult]:
        problems = self.load_gsm8k(num_problems)
        results = []
        
        print("\nRunning Phase 2 experiment (Balanced Strategy with Fixed Injection)...")
        for idx, problem in enumerate(tqdm(problems, desc="Processing problems")):
            try:
                question = problem["question"]
                answer_text = problem["answer"]
                
                gt_match = re.search(r'####\s*(\d+\.?\d*)', answer_text)
                ground_truth = float(gt_match.group(1)) if gt_match else None
                
                # Step 1: Generate original
                original_cot = self.generate_original_cot(question)
                original_answer = self.extract_final_answer(original_cot)
                
                # Step 2: Inject error (Targeting Equations + Newline)
                injection_result = self.inject_error(original_cot)
                if injection_result is None:
                    continue
                
                prefix, orig_val, inj_val, context = injection_result
                
                # === STRATEGY SELECTION ===
                target_faithful_mode = (idx % 2 == 0)
                strategy_name = "forced_faithful" if target_faithful_mode else "standard"
                
                # Step 3: Continue
                prompt = self.create_cot_prompt(question, force_faithful_mode=target_faithful_mode)
                continued_cot = self.continue_generation(prompt, prefix)
                
                # Step 4: Extract final
                full_continued = prefix + continued_cot
                final_answer = self.extract_final_answer(full_continued)
                
                # Step 5: Classify
                is_faithful, notes = self.classify_faithfulness(
                    orig_val, inj_val, continued_cot, final_answer, ground_truth
                )
                
                if idx % 5 == 0:
                    print(f"\n{'='*60}")
                    print(f"[Sample {idx}] Strategy: {strategy_name.upper()}")
                    print(f"Original: {orig_val} -> Injected: {inj_val}")
                    print(f"Context: ...{context.strip()}...")
                    print(f"Prompt Ends With: ...{prefix[-20:].replace(chr(10), '<NL>')}...")
                    print(f"Classification: {'Faithful (Group A)' if is_faithful else 'Stubborn (Group B)' if is_faithful is False else 'Unclear'}")
                    print(f"Notes: {notes}")
                    print(f"{'='*60}\n")
                
                result = GenerationResult(
                    problem_id=idx,
                    problem_text=question,
                    ground_truth_answer=ground_truth,
                    original_cot=original_cot,
                    original_answer=original_answer if original_answer else -1,
                    injection_point=context,
                    original_value=orig_val,
                    injected_value=inj_val,
                    continued_cot=full_continued,
                    final_answer=final_answer if final_answer else -1,
                    is_faithful=is_faithful,
                    prompt_strategy=strategy_name,
                    notes=notes
                )
                
                results.append(result)
                with open(output_file, "a") as f:
                    f.write(json.dumps(asdict(result)) + "\n")
                
            except Exception as e:
                print(f"\nError processing problem {idx}: {e}")
                continue
        
        return results
    
    def analyze_results(self, results: List[GenerationResult]):
        faithful_count = sum(1 for r in results if r.is_faithful is True)
        unfaithful_count = sum(1 for r in results if r.is_faithful is False)
        
        forced_faithful = [r for r in results if r.prompt_strategy == "forced_faithful"]
        standard = [r for r in results if r.prompt_strategy == "standard"]
        ff_success = sum(1 for r in forced_faithful if r.is_faithful is True)
        st_stubborn = sum(1 for r in standard if r.is_faithful is False)
        
        print("\n" + "="*60)
        print("PHASE 2 RESULTS (FINAL)")
        print("="*60)
        print(f"Total problems: {len(results)}")
        print(f"Total Group A (Faithful): {faithful_count}")
        print(f"Total Group B (Stubborn): {unfaithful_count}")
        print("-" * 30)
        print("Strategy Effectiveness:")
        if forced_faithful:
            print(f"Forced Faithful (Few-Shot): {ff_success}/{len(forced_faithful)} ({ff_success/len(forced_faithful)*100:.1f}%)")
        if standard:
            print(f"Standard Prompt:            {st_stubborn}/{len(standard)} ({st_stubborn/len(standard)*100:.1f}%)")
        print("="*60)
        
        if faithful_count >= 10 and unfaithful_count >= 10:
             print("✓ SUCCESS: Ready for Geometry Analysis")
        else:
             print("✗ ADJUSTMENT NEEDED")

def main():
    # Configuration
    MODEL_NAME = "Qwen/Qwen2.5-Math-7B-Instruct" 
    NUM_PROBLEMS = 60
    OUTPUT_FILE = "self_correction_results_phase3_fixed.jsonl"
    
    random.seed(42)
    torch.manual_seed(42)
    
    experiment = SelfCorrectionExperiment(model_name=MODEL_NAME, device="mps") # Change to cuda if needed
    results = experiment.run_experiment(num_problems=NUM_PROBLEMS, output_file=OUTPUT_FILE)
    experiment.analyze_results(results)

if __name__ == "__main__":
    main()