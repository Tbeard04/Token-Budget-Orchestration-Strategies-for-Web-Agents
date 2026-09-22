# Join both A, B and metadata to create single training dataset for C. 
# 1. Load the metadata
# 2. Load the A dataset
# 3. Load the B dataset
# 4. Join the datasets on the task_id column
# 5. Save the joined dataset to a new file in the processed directory
# 6. each row needs to contain both A and B - IMPORTANT 
# split function to allow for different splits of the dataset (for analysis comparing difficulty tiers etc.. between A & B tasks) --> probably a seperate script for this


import json