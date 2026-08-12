"""
Since the AWS instance crashed sometimes or there were errors, re-running the "run_batch.py" resulted in some tasks/budgets being completed later.
i.e. 6 completed budgets for task 1, 
5 completed budgets for task 2, 
instance crashes & re-run batch script, 
6 completed budgets for task 3, 
1 completed budget for task 2.

Sorting will be completed by timestamp. If 1/6 budgets is completed later, then simply move it to the end of the list for that task.
"""

