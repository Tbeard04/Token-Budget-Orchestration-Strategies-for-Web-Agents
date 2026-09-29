from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import wa_env as W

SITES = W.SITES

#Single-site task ids per site, empty sites dropped.
def pools(sites: list | None = None) -> dict:
    return {s: ids for s, ids in W.single_site_tasks(sites or SITES).items() if ids}

#function to get the configs by id
def configs_by_id() -> dict:
    return {c["task_id"]: c for c in W.load_configs() if "task_id" in c}

#function to sample the task ids
def sample_task_ids(n: int, sites: list, seed: int) -> list:
    #Sample up to n tasks per site. Mirrors run_batch.py's sampling.
    p = W.single_site_tasks(sites)
    rng = random.Random(seed)
    ids: list[int] = []
    for site in sites:
        pool = p.get(site, [])
        if pool:
            ids.extend(rng.sample(pool, min(n, len(pool))))
    return sorted(ids)

#function to allocate the tasks to the sites
def allocate(n: int, site_pools: dict) -> dict:
    #Split n across sites in proportion to pool size (largest remainder)
    total = sum(len(v) for v in site_pools.values())
    if not total:
        return {}
    #sort the sites by the pool size
    sites = sorted(site_pools, key=lambda s: -len(site_pools[s]))
    #calculate the exact allocation
    exact = {s: n * len(site_pools[s]) / total for s in sites}
    #calculate the allocation
    alloc = {s: min(int(exact[s]), len(site_pools[s])) for s in sites}
    #calculate the remainder
    left = n - sum(alloc.values())
    for s in sorted(sites, key=lambda s: -(exact[s] - int(exact[s]))):
        if left > 0 and alloc[s] < len(site_pools[s]):
            alloc[s] += 1
            left -= 1

    # a site ran out of tasks: spill the remainder onto whichever still has room
    i = 0
    while left > 0 and i < 10_000:
        s = sites[i % len(sites)]
        if alloc[s] < len(site_pools[s]):
            alloc[s] += 1
            left -= 1
        i += 1
    return alloc