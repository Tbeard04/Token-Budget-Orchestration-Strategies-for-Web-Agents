## AI Usage Introduction

### This document is for AI Usage clarification but also has its own section in references on the report (in Appendix X).

Throughout the development of this project, I utilised AI, specifically Claude's Large Language Model Opus 5 to assist with complex code planning, debugging, and optimising Strategy A, B and C's logic.
The LLM provided structured guidance on functions & code understanding. Most importantly, I didn't just rely on the outputs from the LLM, as I reviewed, tested and iteratively improved them to align with my objectives but to also learn from it's mistakes.

Overall, this allowed me to not only speed development up, but also understand the workings of each system component in much greater depth, bringing planning & design with practical implementation.
The section below will outline where I used Opus 5 and will be structured with the following:

	- Script: xxxx.py
	- Model: Claude Opus 5
	- Prompt: "Write a function that..." 
	- Lines/Output: xx - xx
	- What I've learnt: "I've learnt from this generated output that..."
	- Mistakes Made: "This generated code made a mistake by outputting to a halucinated directory..." | None


Script: label_tool.py
Model: Claude Opus 5
Prompt: add a function to label_tool.py that checks the completed CSV hand labelling gold set file and compares it to the original unedited gold set JSONL to check for any inconsistencies
Lines/Output: 206-250
What I've Learnt: It surfaced a check I hadn't thought to apply to myself: my hand labels had to agree with each other before they could serve as ground truth.
Mistakes Made: None

Script: classify_tasks.py
Model: Claude Opus 5
Prompt: Write a self-test function which includes these 32 selected tasks [attached txt file of tasks]
Lines/Output: 263-345
What I've Learnt: This allowed me to re-use this in other scripts to check the validity of the various scripts such as flag_episodes.py. 
Mistakes Made: None

Script: 
Model: Claude Opus 5
Prompt: 
Lines/Output:
What I've Learnt:
Mistakes Made: 

Script: 
Model: 
Prompt: 
Lines/Output: 
What I've Learnt:  
Mistakes Made: 

Script:
Model: Claude Opus 5
Prompt:
Lines/Output: 
What I've Learnt:
Mistakes Made: 

Script:
Model: Claude Opus 5
Prompt:
Lines/Output: 
What I've Learnt: 
Mistakes Made: 