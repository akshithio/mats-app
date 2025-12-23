
-- Run GSM8K on Qwen & Intervene --

01a.py
01b.py - Introduce Simon Says & Forced Faithfullness
01c.py - Change Simon Says to Few-Shot Prompt & Fix Interjection
01d.py - Removed Forced Faithfulness

-- Do Logit Lens, Attention Attribution, Layer Analysis on 01c.jsonl --

02a.py - Attention patterns (What does the model look at?)
02b.py - Belief state (What does the model represent?)
02c.py - Belief dynamics (How does this evolve through layers?)

03a.py - Probe separability (Can we predict behavior from hidden states?)
03b.py - Truth accessibility (Is the truth "available" at Layer 7?)
03c.py - Truth emergence (When does the truth surface?)

-- Run GSM8K & Interve w/ Noise checking for Rigidity --

04a.py - Gaussian Noise Addition to 01d.py