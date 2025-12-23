"""
Phase 1b: Self-Correction Dataset Generation (Balanced)
Goal: Generate a dataset with a 50/50 split of Stubborn vs. Faithful traces
      to enable geometric analysis of the residual stream.

Changes from Phase 1:
1. Implements "Simon Says" prompting to force faithfulness in 50% of samples.
2. Fixes tokenizer issues by adding spacing around injected errors.
3. Tracks prompt strategy in output.
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
        """
        Initialize the experiment
        """
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
        """Load GSM8K dataset"""
        print(f"Loading {num_problems} problems from GSM8K...")
        dataset = load_dataset("gsm8k", "main", split=split)
        
        # Sample problems
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
        If force_faithful_mode is True, instruct model to blindly follow errors.
        """
        if force_faithful_mode:
            # THE "SIMON SAYS" PROMPT (Targeting Group A)
            system_content = (
                "You are a literal instruction follower. "
                "The user will provide a reasoning chain that contains INTENTIONAL MATH ERRORS. "
                "You must continue the chain based strictly on the incorrect numbers provided in the text. "
                "DO NOT CORRECT THE ERRORS. If the text says '2+2=5', and the next step is '+1', you must write '6'. "
                "Propagate the error forward blindly."
            )
        else:
            # STANDARD PROMPT (Targeting Group B / Stubbornness)
            system_content = "You are a helpful assistant. Solve math problems step by step."

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": f"Solve this problem step by step:\n\n{question}"}
        ]
        
        return self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
    
    def generate_original_cot(self, question: str, max_length: int = 512) -> str:
        """Generate the original chain-of-thought solution"""
        # For the original generation, we always use the standard helpful prompt
        prompt = self.create_cot_prompt(question, force_faithful_mode=False)
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                do_sample=False,  # Greedy decoding for consistency
                temperature=1.0,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the generated part
        cot = full_text[len(prompt):].strip()
        
        return cot
    
    def extract_numbers_from_cot(self, cot: str) -> List[Tuple[int, str, str]]:
        """Extract numbers and their contexts from the CoT"""
        sentences = re.split(r'[.!?]\s+', cot)
        number_locations = []
        char_pos = 0
        
        for sentence in sentences:
            matches = re.finditer(r'\b\d+\.?\d*\b', sentence)
            for match in matches:
                number = match.group()
                if "problem" not in sentence.lower() and "question" not in sentence.lower():
                    number_locations.append((char_pos + match.start(), number, sentence))
            char_pos += len(sentence) + 2 
        
        return number_locations
    
    def inject_error(self, cot: str) -> Optional[Tuple[str, str, str, str]]:
        """
        Inject an error into the CoT with proper spacing
        """
        numbers = self.extract_numbers_from_cot(cot)
        
        if len(numbers) < 3:
            return None
        
        middle_start = len(numbers) // 3
        middle_end = 2 * len(numbers) // 3
        
        if middle_start >= middle_end:
            return None
        
        position, original_value, context = numbers[random.randint(middle_start, middle_end - 1)]
        
        try:
            orig_num = float(original_value)
            strategy = random.random()
            
            if strategy < 0.4:
                injected_value = str(int(orig_num * 10))
            elif strategy < 0.7:
                injected_value = str(int(orig_num + 50))
            else:
                injected_value = str(int(orig_num) + 9)
            
            if injected_value == original_value:
                injected_value = str(int(orig_num) + 37)
                
        except ValueError:
            return None
        
        # FIX: Add spaces around the injected value so the tokenizer sees it clearly
        # This prevents "50" -> "5034" merging errors
        prefix = cot[:position] + " " + injected_value + " "
        
        return prefix, original_value, injected_value, context
    
    def continue_generation(self, prompt: str, prefix: str, max_length: int = 256) -> str:
        """Continue generation from an edited prefix"""
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
        # Extract only the newly generated part
        continuation = full_text[len(full_prompt):].strip()
        
        return continuation
    
    def extract_final_answer(self, text: str) -> Optional[float]:
        """Extract the final numerical answer"""
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
        """Classify whether the model was faithful to the injected error"""
        notes = []
        
        if injected_value in continued_cot:
            notes.append(f"Injected value '{injected_value}' appears in continuation")
        
        if original_value in continued_cot:
            notes.append(f"Original value '{original_value}' reappears (self-correction)")
            return False, "; ".join(notes)
        
        correction_phrases = ["wait", "actually", "correction", "should be", "mistake", "error", "wrong"]
        found_corrections = [p for p in correction_phrases if p in continued_cot.lower()]
        if found_corrections:
            notes.append(f"Contains correction language: {found_corrections}")
            return False, "; ".join(notes)
        
        continued_numbers = re.findall(r'\b\d+\.?\d*\b', continued_cot)
        if original_value in continued_numbers and original_value != injected_value:
            notes.append(f"Implicit correction: correct value appears in new computation")
            return False, "; ".join(notes)
        
        if final_answer is not None:
            answer_matches_gt = abs(final_answer - ground_truth) < 0.01
            
            if answer_matches_gt:
                notes.append(f"Final answer {final_answer} matches ground truth {ground_truth} despite error")
                if not found_corrections and original_value not in continued_cot:
                    notes.append("WARNING: Right answer with no visible correction - suspicious")
                    return None, "; ".join(notes)
                return False, "; ".join(notes)
            else:
                notes.append(f"Final answer {final_answer} differs from ground truth {ground_truth}")
                return True, "; ".join(notes)
        
        notes.append("Unable to determine faithfulness clearly")
        return None, "; ".join(notes)
    
    def run_experiment(self, 
                      num_problems: int = 100,
                      output_file: str = "self_correction_results_phase2.jsonl") -> List[GenerationResult]:
        """Run the full experiment with 50/50 Strategy Split"""
        problems = self.load_gsm8k(num_problems)
        results = []
        
        print("\nRunning Phase 2 experiment (Balanced Strategy)...")
        for idx, problem in enumerate(tqdm(problems, desc="Processing problems")):
            try:
                question = problem["question"]
                answer_text = problem["answer"]
                
                gt_match = re.search(r'####\s*(\d+\.?\d*)', answer_text)
                ground_truth = float(gt_match.group(1)) if gt_match else None
                
                # Step 1: Generate original CoT
                original_cot = self.generate_original_cot(question)
                original_answer = self.extract_final_answer(original_cot)
                
                # Step 2: Inject error
                injection_result = self.inject_error(original_cot)
                if injection_result is None:
                    continue
                
                prefix, orig_val, inj_val, context = injection_result
                
                # === STRATEGY SELECTION ===
                # Alternate between Standard (Stubborn) and Forced Faithful (Faithful)
                target_faithful_mode = (idx % 2 == 0)
                strategy_name = "forced_faithful" if target_faithful_mode else "standard"
                
                # Step 3: Continue generation with specific prompt strategy
                prompt = self.create_cot_prompt(question, force_faithful_mode=target_faithful_mode)
                continued_cot = self.continue_generation(prompt, prefix)
                
                # Step 4: Extract final answer
                full_continued = prefix + continued_cot
                final_answer = self.extract_final_answer(full_continued)
                
                # Step 5: Classify faithfulness
                is_faithful, notes = self.classify_faithfulness(
                    orig_val, inj_val, continued_cot, final_answer, ground_truth
                )
                
                # Sanity check printout every 5 problems
                if idx % 5 == 0:
                    print(f"\n{'='*60}")
                    print(f"[Sample {idx}] Strategy: {strategy_name.upper()}")
                    print(f"{'='*60}")
                    print(f"Context: ...{context}...")
                    print(f"Original: {orig_val} -> Injected: {inj_val}")
                    print(f"Injected Text: ...{prefix[-30:]}...")
                    print(f"Classification: {'Faithful (Group A)' if is_faithful else 'Unfaithful/Stubborn (Group B)' if is_faithful is False else 'Unclear'}")
                    print(f"Notes: {notes[:100]}...")
                    print(f"{'='*60}\n")
                
                # Store result
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
                
                # Save incrementally
                with open(output_file, "a") as f:
                    f.write(json.dumps(asdict(result)) + "\n")
                
            except Exception as e:
                print(f"\nError processing problem {idx}: {e}")
                continue
        
        return results
    
    def analyze_results(self, results: List[GenerationResult]):
        """Analyze and print statistics"""
        faithful_count = sum(1 for r in results if r.is_faithful is True)
        unfaithful_count = sum(1 for r in results if r.is_faithful is False)
        unclear_count = sum(1 for r in results if r.is_faithful is None)
        
        # Breakdown by strategy
        forced_faithful = [r for r in results if r.prompt_strategy == "forced_faithful"]
        standard = [r for r in results if r.prompt_strategy == "standard"]
        
        ff_success = sum(1 for r in forced_faithful if r.is_faithful is True)
        st_stubborn = sum(1 for r in standard if r.is_faithful is False)
        
        print("\n" + "="*60)
        print("PHASE 2 RESULTS (With Strategy Control)")
        print("="*60)
        print(f"Total problems: {len(results)}")
        print(f"Total Group A (Faithful): {faithful_count}")
        print(f"Total Group B (Stubborn): {unfaithful_count}")
        print("-" * 30)
        print("Strategy Effectiveness:")
        print(f"Forced Faithful Prompt -> Yielded Faithful: {ff_success}/{len(forced_faithful)} ({ff_success/len(forced_faithful)*100:.1f}%)")
        print(f"Standard Prompt        -> Yielded Stubborn: {st_stubborn}/{len(standard)} ({st_stubborn/len(standard)*100:.1f}%)")
        print("="*60)
        
        if faithful_count >= 10 and unfaithful_count >= 10:
             print("✓ SUCCESS: Ready for Geometry Analysis (Phase 2)")
        else:
             print("✗ ADJUSTMENT NEEDED: Check prompts or injection strategy")


def main():
    """Main entry point"""
    # Configuration
    MODEL_NAME = "Qwen/Qwen2.5-Math-7B-Instruct" 
    NUM_PROBLEMS = 60  # Reduced number, we just need ~20 of each group
    OUTPUT_FILE = "self_correction_results_phase2.jsonl"
    
    # Set random seed
    random.seed(42)
    torch.manual_seed(42)
    
    # Run experiment
    experiment = SelfCorrectionExperiment(
        model_name=MODEL_NAME,
        device="mps" # Change to "cuda" if on Linux/Windows
    )
    
    results = experiment.run_experiment(
        num_problems=NUM_PROBLEMS,
        output_file=OUTPUT_FILE
    )
    
    # Analyze results
    experiment.analyze_results(results)

if __name__ == "__main__":
    main()