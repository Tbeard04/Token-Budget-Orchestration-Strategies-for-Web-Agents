## Reproducing the Experiment & Environment
This guide rebuilds the AWS environment used to collect every episode in this project (Strategies A, B and C, and the spending-scheme runs). 
It is reconstructed from the project journal. Stock WebArena will not run this project unmodified. 
The WebArena codebase (2023) is incompatible with the current OpenAI SDK and API, and BrowserGym's default action timeout is too short for this deployment. 
The required fixes are listed in Section 0 and applied in Sections 3–5.

### Section 0. Summary of deviations from stock WebArena/BrowserGym
---

| # | Component | Stock behaviour | This project |
|---|---|---|---|
| 1 | WebArena LLM evaluators (`llm_fuzzy_match`, `llm_ua_match`) | Call `gpt-4-1106-preview` | Call `gpt-4o-mini` |
| 2 | WebArena `llms` module | Imports `openai.error` (openai < 1.0) | `openai.error` stubbed at import time in `src/wa_env.py` | Module removed in openai ≥ 1.0; Pydantic AI requires openai 1.x, so downgrading was not an option |
| 3 | WebArena `generate_from_openai_chat_completion` | `openai.ChatCompletion.create` | openai ≥ 1.0 client, same signature, patched at runtime in `src/wa_env.py` |
| 4 | `HTMLContentEvaluator` in `~/webarena/evaluation_harness/evaluators.py` | Assumes `required_contents` is a string | Accepts string or dict |
| 5 | BrowserGym action timeout (`browsergym/core/action/functions.py`) | 500 ms per action | 10000 ms |
| 6 | Magento base URL | Set to the server hostname | `http://localhost:7770` / `:7780` as AWS Instance initialised with no Elastic IP |
| 7 | Sites hosted | 5 sites (+ map, wiki) | Scope on 3 sites: Shopping (7770), Shopping Admin/CMS (7780), Forum/Postmill (9999) |

---

### Section 1. Launch the EC2 Instance on AWS
This follows the WebArena environment README which is located at: <https://github.com/web-arena-x/webarena/blob/main/environment_docker/README.md>

---

| Setting | Value |
|---|---|
| Region | us-east-2 (Ohio), where the WebArena AMI is published |
| AMI | webarena-with-configurable-map-backend — ami-08a862bf98e3bd7aa (Ubuntu 22.04; all WebArena sites pre-installed as Docker containers; Miniconda pre-installed at ~/miniconda3 |
| Instance Type | t3a.xlarge (4 vCPU, 16 GiB RAM) |
| Storage | 1 × 1000 GiB gp3 root volume |
| Key Pair | New key pair webarena-key then download “webarena-key.pem” |
| Elastic IP | None (the public IP changes on each stop/start; look it up in the EC2 console before connecting) |

---

Security group (inbound rules)

---
| Type | Port | Source | Purpose |
|---|---|---|---|
| SSH | 22 | My IP Only | Access to the Instance |
| Custom TCP | 7770 | 0.0.0.0/0 | Shopping (oneStopShop) |
| Custom TCP | 7780 | 0.0.0.0/0 | Shopping Admin (CMS) |
| Custom TCP | 9999 | 0.0.0.0/0 | Forum (Postmill) |
---


Once the instance is creatred, start it from the AWS EC2 Management Concole. Once started, wait for the status check (completed when
3/3 checks passed). Since no Elastic IP is set, check the public IPv4 address to be able to SSH in. 

Open a terminal inside the root folder where the "webarena-key.pem" is located and run:

```bash
chmod 400 webarena-key.pem
ssh -i webarena-key.pem ubuntu@PUBLIC_IPv4
```

### Section 2. Python Environment
Miniconda comes with the AMI, so only the project environment needs creating. On the same terminal once connected to the instance:
```bash
conda create -n tokenbudget python=3.11 -y
conda activate tokenbudget

git clone https://github.com/Tbeard04/Token-Budget-Orchestration-Strategies-for-Web-Agents.git ~/project
cd ~/project
pip install -r requirements_instance.txt
playwright install --with-deps chromium
```

During the Playwright install, Ubuntu shows a "Daemons using outdated libraries" dialog. Accept the pre-ticked defaults (Tab → OK → Enter). 
Leave networkd-dispatcher & unattened-upgrades.service unticked.


### Section 3. Install WebArena’s top level modules
browsergym-webarena installs WebArena's task configs but not its top-level modules (browser_env, llms), which BrowserGym's evaluators import. 
Without this step every task fails with ModuleNotFoundError: No module named 'browser_env', then llms.providers.

Inside conda tokenbudget:
```bash
pip install --force-reinstall --no-deps "webarena @ git+https://github.com/web-arena-x/webarena.git"

cd ~
git clone https://github.com/web-arena-x/webarena.git webarena-src
cd webarena-src && pip install -e . --no-deps && cd ~
```

--no-deps is deliberate. WebArena's own requirements pin 2023 versions (including openai < 1.0) that would break BrowserGym and Pydantic AI.

The remaining import-time failure (import openai.error) is handled in code: src/wa_env.py installs a stub openai.error module before importing browsergym.webarena. 
No manual step is needed. 


### Section 4. Environment variables
The code readsits settings from src/.env. That file is git-ignored because it holds the API key, so create it from the template in the same folder (replace your key at "your-openai-key"):
```bash
cd ~/project/src
cp .env.example .env
sed -i 's/sk-REPLACE_ME/your-openai-key/' .env
```
Localhost and the ports are correct as they are, because the agent runs on the same instance that hosts the sites.

Once you've completed the step above, src/wa_env.py loads this file and maps all the WebArena (WA_*) names onto the bare names WebArena expects (Shopping, Shopping_Admin & Reddit)
before importing browsergym.webarena. The import order matters, because importing first leaves the environment unconfigured.

### Section 5. Apply the WebArena / BrowserGym patches
```bash
conda activate tokenbudget
bash ~/project/setup/patch_webarena.sh
```

#### 5.1 Evaluator Model
`gpt-4-1106-preview` to `gpt-4o-mini` in the installed WebArena package:
- `site-packages/webarena/llms/providers/openai_utils.py`
- `site-packages/webarena/evaluation_harness/helper_functions.py`

#### 5.2 HTMLContentEvaluator dict support
Applied to `~/webarena/evaluation_harness/evaluators.py` during calibration:
```python
#before
clean(x) for x in required_contents.split(" |OR| ")
#after
clean(x) for x in (required_contents if isinstance(required_contents, str)
    else " |OR| ".join(map(str, required_contents.get("must_include", [required_contents.get("exact_match", "")])))).split(" |OR| ")
```

#### 5.3 BrowserGym action timeout
In `site-packages/browsergym/core/action/functions.py`: `timeout=500)` to `timeout=10000)`

#### 5.4 Clear stale bytecode
The script deletes __pycache__ directories under the patched packages so the edits take effect.

### Section 6. Initialising the sites (commands provided by WebArena Environment ReadMe)
<https://github.com/web-arena-x/webarena/blob/main/environment_docker/README.md>

```bash
#1. Recreate the containers from the original images (images are not deleted)
#Magento and its database take ~2 minutes to boot
docker stop shopping shopping_admin forum
docker rm shopping shopping_admin forum
docker run --name shopping -p 7770:80 -d shopping_final_0712
docker run --name shopping_admin -p 7780:80 -d shopping_admin_final_0719
docker run --name forum -p 9999:80 -d postmill-populated-exposed-withimg
sleep 120
```

```bash
#2. Point Magento at localhost (the forum needs no configuration)
docker exec shopping /var/www/magento2/bin/magento setup:store-config:set --base-url="http://localhost:7770"
docker exec shopping mysql -u magentouser -pMyPassword magentodb -e \
  'UPDATE core_config_data SET value="http://localhost:7770/" WHERE path LIKE "web/%base_url";'
docker exec shopping /var/www/magento2/bin/magento cache:flush

docker exec shopping_admin /var/www/magento2/bin/magento setup:store-config:set --base-url="http://localhost:7780"
docker exec shopping_admin mysql -u magentouser -pMyPassword magentodb -e \
  'UPDATE core_config_data SET value="http://localhost:7780/" WHERE path LIKE "web/%base_url";'
docker exec shopping_admin /var/www/magento2/bin/magento cache:flush
```

```bash
#3. Check all three sites respond
curl -s -o /dev/null -w "shopping %{http_code}\n" http://localhost:7770
curl -s -o /dev/null -w "cms %{http_code}\n" http://localhost:7780
curl -s -o /dev/null -w "forum %{http_code}\n" http://localhost:9999
```

Each site should report `200` or `302`. A `302` from Magento is a normal redirect. 
`magentouser` / `MyPassword` are the public defaults shipped in the WebArena AMI.

- If a `docker exec ... mysql` line fails with a connection error, the database is still booting. Wait a minute and run step 2 again.
- The `localhost` setting is stored inside each container, so it survives stopping and starting the instance. *After an AWS restart, only this is needed:*
```bash
docker start shopping shopping_admin forum
```

### Section 7. Strategy Tests
Make sure you are in the conda & in the src directory, if not then:
```bash
conda activate tokenbudget
cd ~/project/src
```

#### 7.1 Strategy A single task and single budget
```bash
python strategy_a.py --budget 32000 276
```

#### 7.2 Strategy B single task and single budget
```bash
python strategy_b.py --budget 32000 276
```

#### 7.3 Strategy C single task and single budget
```bash
python strategy_c.py --budget 32000 276
```

### Section 8. Running the collections
All runs are made from ~project/src inside tmux, so they survive SSH disconnects. Tmux starts in (base), so activate the environment inside it. Run 1 batch at a time as 1 batch takes around 20-30 hours to complete.
```bash
tmux new -s collect
conda activate tokenbudget && cd ~/project/src

python run_batch.py --strategy A --n 999 ||
python run_batch.py --strategy B --n 999 ||
python run_batch.py --strategy C --n 999 --out ../data/raw/6_budgets_ALL_tasks_batch/batch_Strategy_C/strategy_c.jsonl
```

To detatch from the tmux session do: Ctrl-B then D

To re-attach to the tmux session:
```bash
tmux attach -t collect
```

- --n is tasks per site; min(n, pool size) is taken, so --n 999 means every task (all 475 tasks)
- After a run, environment errors (page-load timeouts, Chromium crashes) are removed with python -m clean_contamination.remove_errors --file <jsonl batch file> (from localy machine). Re-running the same run_batch.py command then re-collects only those episodes.

Spending scheme run (100 read only tasks at a fixed budget of 32k tokens):
```bash
python -m spending_schemes.run_schemes 2>&1 | tee -a ../data/processed/spending_schemes/run_log.txt
```

### Section 9. Resetting site states
State-changing tasks (orders, posts, address updates) permanently modify the containers. Before each new collection, the sites were restored by re-running all of Section 6. 
That recreates the containers from the original AMI images and reapplies the localhost setting. 
Within a collection, repeated attempts at the same state-changing task are handled by the decontamination pipeline: remove_errors.py (after removing errors re-run batch script then retrieve results to local machine) --> sort_batch.py --> classify_tasks.py --> flag_episodes.py


### Section 10. Restrieving the results
From the local repo root (Token-Budget-Orchestration-Strategies-for-Web-Agents):
```bash
scp -i keys/webarena-key.pem "ubuntu@PUBLIC_IP:~/project/data/raw/6_budgets_ALL_tasks_batch/batch_Strategy_C/strategy_c*.jsonl" "data/raw/6_budgets_ALL_tasks_batch/batch_Strategy_C/" ||
scp -i keys/webarena-key.pem "ubuntu@PUBLIC_IP:~/project/data/raw/6_budgets_ALL_tasks_batch/batch_Strategy_B/strategy_b.jsonl" "data/raw/6_budgets_ALL_tasks_batch/batch_Strategy_B/"  ||
scp -i keys/webarena-key.pem "ubuntu@PUBLIC_IP:~/project/data/raw/6_budgets_ALL_tasks_batch/batch_Strategy_A/strategy_a.jsonl" "data/raw/6_budgets_ALL_tasks_batch/batch_Strategy_A/"  ||
scp -i keys/webarena-key.pem "ubuntu@PUBLIC_IP:~/project/data/processed/schemes_output/schemes.jsonl" "data/processed/schemes_output/"
```
- Public IP is the AWS Instance IP. 
- The first directory is pulling the data from the instance
- The second directory is where the data goes locally (i.e. in the folder batch_Strategy)


### Section 11. Running the analysis/decontamination/training locally (no AWS required)

#### 11.1 Cloning the repo
```bash
git clone https://github.com/Tbeard04/Token-Budget-Orchestration-Strategies-for-Web-Agents.git
cd Token-Budget-Orchestration-Strategies-for-Web-Agents
```
The data has already been collected/modified by the various modules, but if you decide to run a seperate collection, add an output argument to run_batch.py. (Refer to `Dir_info.md` for more information)

#### 11.2 Create an isolated Python environment (Windows Only)
```bash
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
```
The VScode or windows terminal should now start with a (.venv).

#### 11.3 Install the local requirements
```bash
python -m pip install -r requirements_local.txt
```

#### 11.4 Run the scripts
Refer to `Dir_info.md` in each module.

## Directory Tree:
```text
token-budget-orchestration-strategies-for-web-agents/
│
├── src/
│   ├── wa_env.py           # shared WebArena/BrowserGym harness: task pools, agent construction, budget accounting
│   ├── run_batch.py        # batch driver: runs a strategy over N tasks x 6 budgets
│   ├── strategy_a.py       # Strategy A - single-agent baseline
│   ├── strategy_b.py       # Strategy B - fixed Planner -> Executor -> Critic pipeline
│   ├── .env                # API keys and site URLs (git-ignored)
│   │
│   ├── task_analysis/                  # what the tasks are
│   │   ├── extract_task_intent.py      # intents + regex task_category -> task_intents.json
│   │   ├── task_category_plots.py      # category distribution plots
│   │   └── test.raw.json               # Taken from WebArena repository
│   │
│   ├── task_annotation/                # how hard the tasks are (difficulty rubric)
│   │   ├── annotate_tasks.py           # entry point: gold set, annotation, re-band, merge
│   │   ├── rubric.py                   # the scale: 4 dimensions, prompt, band cut-points
│   │   ├── tasks.py                    # task pools, sampling, site allocation
│   │   ├── observed_cost.py            # per-task navigation cost from a collection file
│   │   ├── gold_set.py                 # blank gold set, exemplar split, merging hand labels
│   │   ├── annotator.py                # the LLM call and the output row
│   │   ├── postprocess.py              # template harmonisation, review flags, re-banding
│   │   ├── reporting.py                # terminal summaries and the validity check
│   │   ├── agreement.py                # Cohen's kappa, Landis & Koch bands, correlations
│   │   └── cohen_kappa.py              # entry point: human-vs-model agreement
│   │
│   ├── clean_contamination/            # decontamination pipeline (ran in this order)
│   │   ├── __init__.py
│   │   ├── remove_errors.py            # 1. strip timeout / environment errors
│   │   ├── sort_batch.py               # 2. sort episodes deterministically
│   │   ├── classify_tasks.py           # 3. read_only / idempotent / non_idempotent
│   │   ├── flag_episodes.py            # 4. per-strategy first-success rule
│   │   └── join_metadata.py            # 5. attach difficulty + category to episodes
│   │
│   └── batch_analysis/                 # results analysis
│       ├── __init__.py
│       ├── shared.py                   # loading, budget bucketing, plot helpers
│       ├── analyse_a.py                # Strategy A cost-performance curves
│       ├── analyse_b.py                # Strategy B, plus critic-invocation analysis
│       ├── compare_ab.py               # A vs B under matched budgets
│       └── compare_b_instructions.py   # B instruction versions v1 / v2 / v3
│
├── data/
│   ├── raw/                                    # collection output, never edited in place
│   │   ├── 6_budgets_ALL_tasks_batch/          # MAIN DATASET - all 475 single-site tasks
│   │   │   ├── batch_Strategy_A/
│   │   │   │   └── strategy_a.jsonl            # 2,850 episodes (475 tasks x 6 budgets)
│   │   │   └── batch_Strategy_B/
│   │   │       └── strategy_b.jsonl
│   │   │
│   │   ├── 6_budgets_201_tasks_batch/          # earlier 67-per-site sample
│   │   │   ├── 67_batch_Strategy_A/
│   │   │   │   ├── strategy_a.jsonl
│   │   │   │   └── analyse_batch_results_terminal_output.txt
│   │   │   └── 67_batch_Strategy_B/
│   │   │       ├── strategy_b.jsonl
│   │   │       └── analyse_batch_results_terminal_output.txt
│   │   │
│   │   ├── pilot_batch_outputs/                # first initial runs
│   │   │   ├── small_batch_test_Strategy_A/
│   │   │   │   ├── pilot_a.jsonl
│   │   │   │   ├── pilot_a_structured.json
│   │   │   │   └── Pilot test batch run for strategy A.txt
│   │   │   └── small_batch_test_Strategy_B/
│   │   │       ├── pilot_b.jsonl
│   │   │       ├── pilot_b_structured.json
│   │   │       └── Pilot test batch run for strategy B.txt
│   │   │
│   │   ├── annotation_tests/                   # difficulty-annotation runs
│   │   │   ├── run1/
│   │   │   │   ├── tasks_annotated_run1.jsonl
│   │   │   │   ├── tasks_annotated_structured_run1.json
│   │   │   │   └── readme.txt                  # with task_intents & A's output, no kappa
│   │   │   └── run2/
│   │   │       └── readme.txt                  # with task_intents & A's output + Cohen's kappa
│   │   │
│   │   ├── strategy_B_instruction_versions/    # prompt provenance for Strategy B
│   │   │   ├── v1_B_instructions.md
│   │   │   ├── v2_B_instructions.md
│   │   │   ├── v3_B_instructions.md
│   │   │   ├── strategy_b_v1_instructions.jsonl
│   │   │   ├── strategy_b_v2_instructions.jsonl
│   │   │   └── strategy_b_v3_instructions.jsonl
│   │   │
│   │   └── readme.txt
│   │
│   └── processed/                              # derived data, reproducible from raw/
│       ├── 6_budgets_ALL_tasks_decontaminated_batch/
│       │   ├── batch_strategy_A/
│       │   └── batch_strategy_B/
│       │
│       ├── task_list/
│       │   └── task_intents.json               # intents + task_category for all 475 tasks from task_analysis/
│       │
│       ├── final_annotation_difficulty_tiers/  # task_metadata.jsonl (final labels)
│       │
│       ├── diagrams/
│       │   ├── 6_budgets_ALL_tasks_decontaminated_batches/
│       │   │   ├── strategy_a_batch/
│       │   │   ├── strategy_b_batch/
│       │   │   └── comparisons/
│       │   ├── difficulty_tiers/
│       │   ├── pilot_C_batch/
│       │   ├── task_list/
│       │   │   ├── task_categories_overall.png
│       │   │   ├── task_categories_by_site.png
│       │   │   └── task_categories_risk.png
│       │   └── other_diagrams/
│       │       ├── 6_budgets_201_tasks/
│       │       │   ├── strategy_a_batch/
│       │       │   │   ├── cost_curve_a.png
│       │       │   │   ├── success_by_site_a.png
│       │       │   │   ├── termination_reasons_a.png
│       │       │   │   └── token_boxplot_a.png
│       │       │   └── strategy_b_batch/
│       │       │       ├── cost_curve_b.png
│       │       │       ├── success_by_site_b.png
│       │       │       ├── termination_reasons_b.png
│       │       │       └── token_boxplot_b.png
│       │       └── pilot_batch/
│       │           ├── pilot_batch_strategy A/
│       │           │   ├── cost_curve_a.png
│       │           │   ├── success_by_site_a.png
│       │           │   ├── termination_reasons_a.png
│       │           │   └── token_boxplot_a.png
│       │           └── pilot_batch_strategy B/
│       │               ├── cost_curve_b.png
│       │               ├── success_by_site_b.png
│       │               ├── termination_reasons_b.png
│       │               └── token_boxplot_b.png
│       │
│       └── readme.txt
│
├── docs/
│   ├── Additional Material/
│   └── Source Code.txt
│
├── keys/
│   └── webarena-key.pem               # AWS SSH key (git-ignored - never commit)
│
├── .venv/
├── .vscode/
├── requirements.txt
├── .gitignore
└── README.md
```
