## AI Usage Introduction

### This document is for AI Usage clarification but also has its own section in references on the report (in Appendix X).

Throughout the development of this project, I utilised AI, specifically Claude's Large Language Model Opus 5 to assist with complex code planning, debugging, and optimising scripts.
The LLM provided structured guidance on functions & code understanding. Most importantly, I didn't just rely on the outputs from the LLM, as I reviewed, tested and iteratively improved them to align with my objectives but to also learn from it's mistakes.

Overall, this allowed me to not only speed development up, but also understand the workings of each system component in much greater depth, bringing planning & design with practical implementation.
The section below will outline where I used Opus 5 and will be structured with the following:

	- Script: xxxx.py
	- Model: Claude Opus 5
	- Prompt: "Write a function that..." 
	- Lines/Output: xx - xx
	- What I've learnt: "I've learnt from this generated output that..."
	- Mistakes Made: "This generated code made a mistake by outputting to a halucinated directory..." | None

---

```text
Script: label_tool.py
Model: Claude Opus 5
Prompt: add a function to label_tool.py that checks the completed CSV hand labelling gold set file and compares it to the original unedited gold set JSONL to check for any inconsistencies
Lines/Output: 206-250
What I've Learnt: It surfaced a check I hadn't thought to apply to myself: my hand labels had to agree with each other before they could serve as ground truth.
Mistakes Made: None
```

```text
Script: classify_tasks.py
Model: Claude Opus 5
Prompt: Write a self-test function which includes these 32 selected tasks [attached txt file of tasks]
Lines/Output: 263-345
What I've Learnt: This allowed me to develop similar functions in other scripts to check the validity such as in flag_episodes.py
Mistakes Made: None
```

```text
Script: shared.py
Model: Claude Opus 5
Prompt: Write a function or 2 if needed that first calculates at least a 95% confidence range for a success rate, given the number of successes and the total number of attempts. It needs to work well when the success rate is very low or zero. And I also need to test whether 1 strategy is genuinely more successful than another when both were run on the same tasks. It should only look at the tasks where one strategy succeeded and the other failed.
Lines/Output: 75-84 and 90-97
What I've Learnt: I learnt that the usual "±1.96 × standard error" interval fails at low success rates. It can drop below 0% and collapses to zero width when there are no successes, which is common at 2k–4k budgets. The Wilson interval stays within 0–1 and stays sensible near zero. From McNemar I learnt that when strategies run on the same tasks, only the tasks where they disagree carry any evidence about which is better. Treating the results as two independent samples would ignore that pairing and give the wrong p-values.
Mistakes Made: None
```

```text
Script: compare_abc.py
Model: Claude Opus 5
Prompt: My analysis script compare_abc.py compares three web-agent strategies (A, B and C) using exact McNemar tests on paired (task, budget) cells. It runs the comparison for each strategy pair (A vs B, A vs C, B vs C), both pooled across budgets and within each difficulty tier (Easy, Medium, Hard). Because this involves many tests at once, apply a correction method.
Lines/Output: 123-150 & 291-313
What I've Learnt: I learnt about the multiple comparisons problem. Running many tests at the 5% level makes it likely that at least one comes out "significant" by chance alone. Dividing the threshold by the number of tests (0.05/3 for the pooled pairs, 0.05/9 within tiers) means only differences too large to be luck are claimed. I also learnt that Bonferroni is conservative, so a borderline result like A vs C (p = 0.065) should be reported as no detectable difference rather than as evidence that the two are equal.
Mistakes Made: None
```

```text
Script: analyse_schemes.py
Model: Claude Opus 5
Prompt: write two functions that compare my results task by task. The first compares each pair of spending schemes within each strategy, testing both success and tokens spent. The second compares the strategies against each other under the same scheme.
Lines/Output: 114-147 & 150-171
What I've Learnt: I learnt that success and cost need different paired tests. So 1st for success as whether yes/no per task, so it uses McNemar on the tasks where the two schemes disagree. And 2nd for tokens which are continious, so they use the Wilcoxon signed rank test on each tasks difference, which doesnt assume a normal distribution. 
Mistakes Made: None
```