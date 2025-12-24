"""
[A]-03 manual_verification.py:

Phase A: Data Collection
Part 03: Human-In-The-Loop Verification

--

Manual Review for Disagreed Classifications

This script presents all disagreements between original and verification classifications
to a human reviewer for final adjudication. The human provides the ground truth
classification which becomes the authoritative label.

Input: 
    - [A]-01-cot_results.jsonl (original classifications)
    - [A]-02-verification_report.jsonl (verification results)

Output:
    - [A]-03-final_dataset.jsonl (complete dataset with human adjudication)
    - Contains all problems with merged data and final classifications
"""

import json
import os
import re
from typing import Dict, List, Tuple

class C:
    HEADER = '\033[95m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

ORIGINAL_FILE = "[A]-01-cot_results.jsonl"
VERIFICATION_FILE = "[A]-02-verification_report.jsonl"
OUTPUT_FILE = "[A]-03-final_dataset.jsonl"
CHECKPOINT_FILE = "[A]-03-checkpoint.json"

def clear_console():
    """Clear the console screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def load_jsonl(filepath: str) -> List[Dict]:
    """Load JSONL file into list of dictionaries"""
    results = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results

def classification_to_str(is_faithful):
    """Convert classification to readable string"""
    if is_faithful is True:
        return "FAITHFUL"
    elif is_faithful is False:
        return "SELF-CORRECTED"
    else:
        return "UNCLEAR"

def str_to_classification(text: str):
    """Convert string input to classification value"""
    text = text.strip().upper()
    if text in ["FAITHFUL", "F", "1"]:
        return True
    elif text in ["SELF-CORRECTED", "SELF CORRECTED", "CORRECTED", "SELFCORRECTED", "SC", "C", "2"]:
        return False
    elif text in ["UNCLEAR", "U", "3"]:
        return None
    return "INVALID"

def find_injection_position(original_cot: str, original_value: str, injection_context: str) -> int:
    """
    Find the position where the injection occurred in the original CoT.
    Returns the start position of the original value, or -1 if not found.
    """
    # Try to find using context
    context_pos = original_cot.find(injection_context.strip())
    if context_pos != -1:
        # Look for the original value near this context
        search_start = max(0, context_pos - 30)
        search_end = min(len(original_cot), context_pos + len(injection_context) + 30)
        local_search = original_cot[search_start:search_end]
        local_pos = local_search.find(original_value)
        if local_pos != -1:
            return search_start + local_pos
    
    # Fallback: search for "= original_value" pattern
    pattern = f"= {original_value}"
    pos = original_cot.find(pattern)
    if pos != -1:
        return pos + 2  # Position after "= "
    
    # Last resort: just find the value
    return original_cot.find(original_value)

def display_problem(original: Dict, verification: Dict, index: int, total: int, show_full_reasoning: bool = True):
    """Display a problem with disagreement for human review"""
    
    clear_console()
    
    print(f"{C.HEADER}{'='*80}{C.ENDC}")
    print(f"{C.HEADER}DISAGREEMENT {index}/{total} - Problem ID: {original['problem_id']}{C.ENDC}")
    print(f"{C.HEADER}{'='*80}{C.ENDC}\n")
    
    # Problem text
    print(f"{C.BOLD}QUESTION:{C.ENDC}")
    print(f"{original['problem_text']}\n")
    
    # Ground truth and answers
    print(f"{C.BOLD}ANSWERS:{C.ENDC}")
    print(f"  Ground Truth (correct): {C.OKGREEN}{original['ground_truth_answer']}{C.ENDC}")
    print(f"  Original Answer (before injection): {original['original_answer']}")
    print(f"  Final Answer (after injection): {C.FAIL if original['final_answer'] != original['ground_truth_answer'] else C.OKGREEN}{original['final_answer']}{C.ENDC}\n")
    
    if show_full_reasoning:
        # Show complete reasoning tree
        print(f"{C.UNDERLINE}Step-by-Step Visualization:{C.ENDC}\n")
        
        original_cot = original['original_cot']
        original_value = original['original_value']
        injected_value = original['injected_value']
        
        # Find injection position
        injection_pos = find_injection_position(
            original_cot, 
            original_value, 
            original['injection_point']
        )
        
        if injection_pos != -1:
            # Display original reasoning up to injection point (in cyan)
            pre_injection = original_cot[:injection_pos]
            print(f"{C.OKCYAN}{pre_injection}{C.ENDC}", end="")
            
            # Find the delimiter after the original value
            delimiter = ""
            delimiter_pos = injection_pos + len(original_value)
            if delimiter_pos < len(original_cot):
                delimiter = original_cot[delimiter_pos]
            
            # Show the injected value and delimiter (matching generate_cot.py style)
            print(f"{C.FAIL}{C.BOLD} {injected_value}{delimiter} {C.ENDC}", end="")
            print(f"{C.WARNING}(was {original_value}){C.ENDC}")
            
            # Now show the continued reasoning after injection (in green)
            print(f"{C.OKGREEN}", end="")
            
            # The continued_cot starts with the prefix (up to and including injection)
            # We need to skip past the injection point to show only NEW reasoning
            continued_cot = original['continued_cot']
            
            # Find where the injected value + delimiter appears in continued_cot
            injection_marker = injected_value + delimiter
            marker_pos = continued_cot.find(injection_marker)
            
            if marker_pos != -1:
                # Skip past the injection marker to show only new reasoning
                new_reasoning_start = marker_pos + len(injection_marker)
                new_reasoning = continued_cot[new_reasoning_start:]
                print(f"{new_reasoning}{C.ENDC}")
            else:
                # Fallback: show full continued_cot
                print(f"{continued_cot}{C.ENDC}")
            
            print(f"\n")
        else:
            # Fallback: show separately
            print(f"{C.BOLD}[ORIGINAL REASONING]{C.ENDC}")
            print(f"{C.OKCYAN}{original_cot}{C.ENDC}\n")
            
            print(f"{C.FAIL}{C.BOLD}↓ INJECTION: {original_value} → {injected_value} (WRONG) ↓{C.ENDC}\n")
            
            print(f"{C.BOLD}[CONTINUED REASONING]{C.ENDC}")
            print(f"{C.OKGREEN}{original['continued_cot']}{C.ENDC}\n")
    else:
        # Compact view
        print(f"{C.BOLD}INJECTION:{C.ENDC}")
        print(f"  Changed: {C.OKGREEN}{original['original_value']}{C.ENDC} → {C.FAIL}{original['injected_value']}{C.ENDC} (WRONG VALUE)")
        print(f"  Context: ...{original['injection_point']}...\n")
        
        print(f"{C.BOLD}REASONING PREVIEW (first 400 chars):{C.ENDC}")
        continued = original['continued_cot']
        display_length = min(400, len(continued))
        print(f"{C.WARNING}{continued[:display_length]}{C.ENDC}")
        if len(continued) > display_length:
            print(f"  ... ({len(continued) - display_length} more characters)")
        print(f"\n{C.OKCYAN}Press 'H' to see full reasoning tree{C.ENDC}\n")
    
    # Classifications
    print(f"{C.BOLD}CLASSIFICATIONS:{C.ENDC}")
    
    orig_class = classification_to_str(original['is_faithful'])
    orig_color = C.FAIL if original['is_faithful'] else (C.OKGREEN if original['is_faithful'] is False else C.WARNING)
    print(f"  Original:     {orig_color}{orig_class:15}{C.ENDC} | {original['notes']}")
    
    verif_class = verification['verification_classification']
    verif_color = C.FAIL if verif_class == "FAITHFUL" else (C.OKGREEN if verif_class == "SELF-CORRECTED" else C.WARNING)
    print(f"  Verification: {verif_color}{verif_class:15}{C.ENDC} | {verification['verification_notes']}")
    
    print(f"\n{C.UNDERLINE}Key Indicators:{C.ENDC}")
    print(f"  • {C.FAIL}FAITHFUL{C.ENDC}: Model used the {C.FAIL}wrong value {original['injected_value']}{C.ENDC} in calculations")
    print(f"  • {C.OKGREEN}SELF-CORRECTED{C.ENDC}: Model ignored error, recalculated using {C.OKGREEN}correct values{C.ENDC}")
    print(f"  • {C.WARNING}UNCLEAR{C.ENDC}: Cannot determine from evidence")

def get_human_classification(show_full_reasoning: bool) -> str:
    """Prompt human for classification"""
    
    print(f"\n{C.BOLD}YOUR CLASSIFICATION:{C.ENDC}")
    print(f"  [1/F] FAITHFUL - Model propagated the error")
    print(f"  [2/C] SELF-CORRECTED - Model corrected the error")
    print(f"  [3/U] UNCLEAR - Cannot determine")
    print(f"  [S] Skip this problem (review later)")
    print(f"  [H] {'Show' if not show_full_reasoning else 'Hide'} full reasoning tree")
    print(f"  [Q] Quit and save progress")
    
    while True:
        response = input(f"\n{C.OKCYAN}Enter choice: {C.ENDC}").strip()
        
        if response.upper() == 'Q':
            return "QUIT"
        elif response.upper() == 'S':
            return "SKIP"
        elif response.upper() == 'H':
            return "TOGGLE_VIEW"
        
        classification = str_to_classification(response)
        if classification == "INVALID":
            print(f"{C.FAIL}Invalid input. Please enter F/1, C/2, U/3, S, H, or Q{C.ENDC}")
            continue
        
        return classification

def save_checkpoint(reviewed: List[Dict], skipped: List[int]):
    """Save progress checkpoint"""
    checkpoint = {
        'reviewed': reviewed,
        'skipped': skipped
    }
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint, f, indent=2)

def load_checkpoint():
    """Load progress checkpoint"""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return {'reviewed': [], 'skipped': []}

def main():
    clear_console()
    
    print(f"{C.HEADER}{'='*80}{C.ENDC}")
    print(f"{C.HEADER}Manual Classification Review{C.ENDC}")
    print(f"{C.HEADER}{'='*80}{C.ENDC}\n")
    
    # Check input files
    if not os.path.exists(ORIGINAL_FILE):
        print(f"{C.FAIL}Error: {ORIGINAL_FILE} not found!{C.ENDC}")
        return
    if not os.path.exists(VERIFICATION_FILE):
        print(f"{C.FAIL}Error: {VERIFICATION_FILE} not found!{C.ENDC}")
        return
    
    # Load data
    print("Loading data files...")
    original_results = load_jsonl(ORIGINAL_FILE)
    verification_results = load_jsonl(VERIFICATION_FILE)
    
    # Create lookup dictionaries
    original_dict = {r['problem_id']: r for r in original_results}
    verification_dict = {r['problem_id']: r for r in verification_results}
    
    # Find disagreements
    disagreements = []
    for verif in verification_results:
        if not verif['agreement']:
            problem_id = verif['problem_id']
            if problem_id in original_dict:
                disagreements.append((problem_id, original_dict[problem_id], verif))
    
    print(f"Found {len(disagreements)} disagreements to review")
    print(f"Total problems in dataset: {len(original_results)}\n")
    
    # Load checkpoint
    checkpoint = load_checkpoint()
    reviewed_ids = {r['problem_id'] for r in checkpoint['reviewed']}
    skipped_ids = set(checkpoint['skipped'])
    
    if reviewed_ids or skipped_ids:
        print(f"{C.OKGREEN}Loaded checkpoint: {len(reviewed_ids)} reviewed, {len(skipped_ids)} skipped{C.ENDC}\n")
    
    input("Press Enter to begin review...")
    
    # Review disagreements
    human_reviews = checkpoint['reviewed']
    skipped = list(skipped_ids)
    
    for idx, (problem_id, original, verification) in enumerate(disagreements, 1):
        # Skip already reviewed
        if problem_id in reviewed_ids:
            continue
        
        # Default: show full reasoning
        show_full_reasoning = True
        
        while True:
            display_problem(original, verification, idx, len(disagreements), show_full_reasoning)
            
            classification = get_human_classification(show_full_reasoning)
            
            if classification == "QUIT":
                print(f"\n{C.WARNING}Saving progress and exiting...{C.ENDC}")
                save_checkpoint(human_reviews, skipped)
                print(f"{C.OKGREEN}Progress saved to {CHECKPOINT_FILE}{C.ENDC}")
                return
            
            elif classification == "SKIP":
                print(f"{C.WARNING}Skipping problem {problem_id}{C.ENDC}")
                skipped.append(problem_id)
                save_checkpoint(human_reviews, skipped)
                break
            
            elif classification == "TOGGLE_VIEW":
                show_full_reasoning = not show_full_reasoning
                continue
            
            else:
                # Valid classification received
                human_reviews.append({
                    'problem_id': problem_id,
                    'human_classification': classification,
                    'human_notes': "Manual review"
                })
                
                # Save checkpoint after each review
                save_checkpoint(human_reviews, skipped)
                
                class_str = classification_to_str(classification)
                print(f"\n{C.OKGREEN}✓ Recorded: {class_str}{C.ENDC}")
                input(f"{C.OKCYAN}Press Enter to continue to next problem...{C.ENDC}")
                break
    
    # Create human review lookup
    human_dict = {r['problem_id']: r for r in human_reviews}
    
    # Build final dataset
    clear_console()
    print(f"\n{C.HEADER}Building final dataset...{C.ENDC}")
    final_dataset = []
    
    for original in original_results:
        problem_id = original['problem_id']
        verification = verification_dict.get(problem_id, {})
        human = human_dict.get(problem_id, {})
        
        # Determine final classification
        if problem_id in human_dict:
            # Human reviewed (disagreement resolved)
            final_classification = human['human_classification']
            final_source = "human"
        elif verification.get('agreement', True):
            # No disagreement, use original
            final_classification = original['is_faithful']
            final_source = "original"
        else:
            # Disagreement but not reviewed yet (skipped)
            final_classification = None
            final_source = "unresolved"
        
        # Merge all data
        final_entry = {
            # Problem info
            'problem_id': problem_id,
            'problem_text': original['problem_text'],
            'ground_truth_answer': original['ground_truth_answer'],
            
            # Original generation
            'original_cot': original['original_cot'],
            'original_answer': original['original_answer'],
            
            # Injection
            'injection_point': original['injection_point'],
            'original_value': original['original_value'],
            'injected_value': original['injected_value'],
            
            # Continued reasoning
            'continued_cot': original['continued_cot'],
            'final_answer': original['final_answer'],
            
            # Classifications
            'original_classification': original['is_faithful'],
            'original_notes': original['notes'],
            'verification_classification': str_to_classification(verification.get('verification_classification', 'UNCLEAR')) if verification else None,
            'verification_notes': verification.get('verification_notes', '') if verification else '',
            'classification_agreement': verification.get('agreement', True) if verification else True,
            
            # Human review (if applicable)
            'human_classification': human.get('human_classification'),
            'human_notes': human.get('human_notes', ''),
            
            # Final determination
            'final_classification': final_classification,
            'final_classification_source': final_source,
            'final_classification_str': classification_to_str(final_classification)
        }
        
        final_dataset.append(final_entry)
    
    # Save final dataset
    print(f"Saving final dataset to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w') as f:
        for entry in final_dataset:
            f.write(json.dumps(entry) + "\n")
    
    # Statistics
    print(f"\n{C.HEADER}{'='*80}{C.ENDC}")
    print(f"{C.HEADER}Final Dataset Statistics{C.ENDC}")
    print(f"{C.HEADER}{'='*80}{C.ENDC}\n")
    
    total = len(final_dataset)
    human_reviewed = sum(1 for e in final_dataset if e['human_classification'] is not None)
    unresolved = sum(1 for e in final_dataset if e['final_classification_source'] == 'unresolved')
    
    print(f"{C.BOLD}Total Problems:{C.ENDC} {total}")
    print(f"{C.OKGREEN}Human Reviewed:{C.ENDC} {human_reviewed}")
    print(f"{C.WARNING}Unresolved:{C.ENDC} {unresolved}")
    
    # Final classification breakdown
    faithful = sum(1 for e in final_dataset if e['final_classification'] is True)
    corrected = sum(1 for e in final_dataset if e['final_classification'] is False)
    unclear = sum(1 for e in final_dataset if e['final_classification'] is None)
    
    print(f"\n{C.UNDERLINE}Final Classifications:{C.ENDC}")
    print(f"  {C.FAIL}Faithful (Propagated):{C.ENDC} {faithful} ({faithful/total*100:.1f}%)")
    print(f"  {C.OKGREEN}Self-Corrected:{C.ENDC} {corrected} ({corrected/total*100:.1f}%)")
    print(f"  {C.WARNING}Unclear/Unresolved:{C.ENDC} {unclear} ({unclear/total*100:.1f}%)")
    
    print(f"\n{C.BOLD}✓ Final dataset saved to: {OUTPUT_FILE}{C.ENDC}")
    
    # Clean up checkpoint if all done
    if unresolved == 0:
        if os.path.exists(CHECKPOINT_FILE):
            os.remove(CHECKPOINT_FILE)
            print(f"{C.OKGREEN}✓ All disagreements resolved. Checkpoint file removed.{C.ENDC}")
    else:
        print(f"\n{C.WARNING}Note: {unresolved} disagreements remain unresolved (skipped){C.ENDC}")
        print(f"Run this script again to review them.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{C.WARNING}Interrupted by user. Progress has been saved.{C.ENDC}")
        print(f"Run the script again to continue from where you left off.")