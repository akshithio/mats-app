# [A] - Data Collection

Notes:

- There were a lot of times where if the data being inserted was of form "1822", then when the model propagated the error, it would use the value as 1820 instead. Some of these examples were used in the final dataset (I would assume), but a good amount weren't.

- Issues with injection made the data collection less efficient than it could've been. There definitely were cases where like If I was trying to replace `1.40` with `2` it would end up being like `2 .40` or like If I was replacing `45 miles per hour` with `82 miles per hour` then it could end up being like `82 miles per hours 45` in which case, the model tended to self-correct itself. 

- In general, a decent number of the times where a model seemed to self-correct itself, it felt like it could be attributed to some kind of issue with the injection logic which if you think about is not a great sign.

- Regardless, there seemed to be a *hopefully* significant number of cases where the model's self-correction was genuine despite proper injection, and is *hopefully* where both the initial LLM and the verifier LLM agreed, so using only these cases should hopefully provide good enough data for Phase 2.

- This entire process, also has me under the impression that I can effectively predict when a CoT intervention will be causal in the next reasoning steps. In another words, that I will know by looking at the CoT upto the injection point and a couple words after, of whether or not the model will self-correct or be faithful and let the injection propagate. 

- I'm not sure if the above is a validated hypothesis but feels like something I could prove by like doing injections right before a newline character (so that the model when it moves to the next line and looks at the previous line has the injected thought) or replacing full reasoning "steps" in the human sense. I could compare this to like a more random injection and then compare the final answer, especially in places where reasoning is important, like the math we are doing here.