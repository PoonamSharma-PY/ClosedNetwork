# # random_pmcsn.py

# import os
# import math
# import time
# import json
# import random
# import numpy as np
# import pandas as pd
# import networkx as nx
# from joblib import Parallel, delayed
# from numba import njit
# from tqdm import tqdm
# import ast

# # --- Configuration ---
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"

# NUM_CPUS = 24
# SIMULATIONS = 10000
# GRAPH_FILE = "euemail_trivalency.txt"
# OUT_CSV = "Random_PMCSN.csv"
# OUT_JSON = "Random_PMCSN.json"

# K_LIST = [5, 10, 15, 20, 25, 30, 35, 40]
# L_LIST = [4, 8, 12, 16, 20, 24, 28]

# # --- Numba ICM ---
# @njit
# def simulate_icm_numba(seed_set, adj_matrix, prob_matrix):
#     n = adj_matrix.shape[0]
#     activated = np.zeros(n, dtype=np.bool_)
#     newly_activated = np.zeros(n, dtype=np.bool_)
#     for node in seed_set:
#         activated[node] = True
#         newly_activated[node] = True
#     steps = 0
#     while np.any(newly_activated):
#         next_new = np.zeros(n, dtype=np.bool_)
#         for u in range(n):
#             if newly_activated[u]:
#                 for j in range(adj_matrix.shape[1]):
#                     v = adj_matrix[u, j]
#                     if v == -1:
#                         break
#                     if not activated[v] and np.random.rand() < prob_matrix[u, j]:
#                         activated[v] = True
#                         next_new[v] = True
#         newly_activated = next_new
#         steps += 1
#     return activated, steps, np.sum(activated)

# def simulate_parallel(seed_set, adj_matrix, prob_matrix, sims):
#     return Parallel(n_jobs=NUM_CPUS)(
#         delayed(simulate_icm_numba)(np.array(seed_set, dtype=np.int32), adj_matrix, prob_matrix)
#         for _ in range(sims)
#     )

# # --- Helpers ---
# def to_matrix(G):
#     n = max(G.nodes()) + 1
#     out_deg = dict(G.out_degree())
#     max_deg = max(out_deg.values(), default=0)
#     adj = -np.ones((n, max_deg), dtype=np.int32)
#     prob = np.zeros((n, max_deg), dtype=np.float32)
#     deg = np.zeros(n, dtype=np.int32)
#     for u, v, data in G.edges(data=True):
#         idx = deg[u]
#         adj[u, idx] = v
#         prob[u, idx] = float(data.get("prob", 0.1))
#         deg[u] += 1
#     return adj, prob

# def build_random_limited_graph(G, seed_set, l):
#     visited = set(seed_set)
#     frontier = list(seed_set)
#     GD = nx.DiGraph()
#     GD.add_nodes_from(G.nodes())

#     while frontier:
#         next_frontier = []
#         for u in frontier:
#             nbrs = list(G.successors(u))
#             selected = nbrs if len(nbrs) <= l else random.sample(nbrs, l)
#             for v in selected:
#                 if G.has_edge(u, v):
#                     GD.add_edge(u, v, prob=G[u][v]["prob"])
#                     if v not in visited:
#                         next_frontier.append(v)
#                         visited.add(v)
#         frontier = next_frontier

#     return GD

# def compute_benefit(activated_mask, benefits_array):
#     return float(np.sum(benefits_array[activated_mask]))

# def load_costs_benefits(cost_path="cost.txt", benefit_path="benefit.txt"):
#     with open(cost_path) as f:
#         costs = ast.literal_eval(f.read())
#     with open(benefit_path) as f:
#         benefits = ast.literal_eval(f.read())
#     return {int(k): float(v) for k, v in costs.items()}, {int(k): float(v) for k, v in benefits.items()}

# def append_rows_csv(path, rows):
#     if not rows: return
#     df = pd.DataFrame(rows)
#     file_exists = os.path.exists(path)
#     df.to_csv(path, mode='a', header=not file_exists, index=False)

# # --- Algorithm ---
# def run_random_pmcsn(G, k, l, costs, benefits):
#     total_start = time.perf_counter()

#     seed_set = random.sample(list(G.nodes()), k)
#     GD = build_random_limited_graph(G, seed_set, l)

#     adj_matrix, prob_matrix = to_matrix(GD)
#     n = adj_matrix.shape[0]
#     benefits_array = np.zeros(n, dtype=np.float32)
#     for node, val in benefits.items():
#         if 0 <= node < n:
#             benefits_array[node] = val

#     sims = simulate_parallel(seed_set, adj_matrix, prob_matrix, sims=SIMULATIONS)
#     benefits_list = [compute_benefit(sim[0], benefits_array) for sim in sims]
#     steps_list = [sim[1] for sim in sims]
#     activations = [sim[2] for sim in sims]

#     avg_benefit = float(np.mean(benefits_list))
#     avg_steps = float(np.mean(steps_list))
#     avg_activated = float(np.mean(activations))
#     seed_cost = float(sum(costs.get(v, 0.0) for v in seed_set))
#     profit = avg_benefit - seed_cost

#     return {
#         "Model Name": "Random PMCSN",
#         "Seed Set": seed_set,
#         "Seed Set Size": len(seed_set),
#         "Seed Set Cost": seed_cost,
#         "Benefit": avg_benefit,
#         "Profit Earned": profit,
#         "Average Timestep": avg_steps,
#         "Average Activated Nodes": avg_activated,
#         "Value of l": l,
#         "Value of k": k,
#         "Total Nodes": GD.number_of_nodes(),
#         "Total Edges": GD.number_of_edges(),
#         "Execution Time (s)": float(time.perf_counter() - total_start)
#     }

# # --- Runner ---
# def main():
#     if os.path.exists(OUT_CSV):
#         os.remove(OUT_CSV)

#     G = nx.read_edgelist(GRAPH_FILE, create_using=nx.DiGraph(), nodetype=int, data=(("prob", float),))
#     costs, benefits = load_costs_benefits()
#     results = []

#     for k, l in tqdm([(k, l) for k in K_LIST for l in L_LIST], desc="(k,l) grid"):
#         row = run_random_pmcsn(G, k, l, costs, benefits)
#         results.append(row)
#         append_rows_csv(OUT_CSV, [row])

#     with open(OUT_JSON, "w") as f:
#         json.dump(results, f, indent=2)

#     print(f"\n✅ Done.\nCSV: {OUT_CSV}\nJSON: {OUT_JSON}")

# if __name__ == "__main__":
#     main()



# random_pmcsn_budget.py

import os
import math
import time
import json
import random
import numpy as np
import pandas as pd
import networkx as nx
from joblib import Parallel, delayed
from numba import njit
from tqdm import tqdm
import ast

# --- Configuration ---
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

NUM_CPUS = 24
SIMULATIONS = 10000
GRAPH_FILE = "euemail_trivalency.txt"
OUT_CSV = "Random_PMCSN_Bl.csv"
OUT_JSON = "Random_PMCSN_Bl.json"

BUDGET_LIST = [500, 1000, 1500, 2000, 2500]
L_LIST = [4, 8, 12, 16, 20, 24, 28]

# --- Numba ICM ---
@njit
def simulate_icm_numba(seed_set, adj_matrix, prob_matrix):
    n = adj_matrix.shape[0]
    activated = np.zeros(n, dtype=np.bool_)
    newly_activated = np.zeros(n, dtype=np.bool_)
    for node in seed_set:
        if 0 <= node < n:
            activated[node] = True
            newly_activated[node] = True
    steps = 0
    while np.any(newly_activated):
        next_new = np.zeros(n, dtype=np.bool_)
        for u in range(n):
            if newly_activated[u]:
                for j in range(adj_matrix.shape[1]):
                    v = adj_matrix[u, j]
                    if v == -1:
                        break
                    if not activated[v] and np.random.rand() < prob_matrix[u, j]:
                        activated[v] = True
                        next_new[v] = True
        newly_activated = next_new
        steps += 1
    return activated, steps, np.sum(activated)

def simulate_parallel(seed_set, adj_matrix, prob_matrix, sims):
    seed_arr = np.array(seed_set, dtype=np.int32)
    return Parallel(n_jobs=NUM_CPUS)(
        delayed(simulate_icm_numba)(seed_arr, adj_matrix, prob_matrix)
        for _ in range(sims)
    )

# --- Helpers ---
def to_matrix(G):
    if G.number_of_nodes() == 0:
        return np.zeros((0,0), dtype=np.int32), np.zeros((0,0), dtype=np.float32)
    n = max(G.nodes()) + 1
    out_deg = dict(G.out_degree())
    max_deg = max(out_deg.values(), default=0)
    if max_deg == 0:
        return -np.ones((n,1), dtype=np.int32), np.zeros((n,1), dtype=np.float32)
    adj = -np.ones((n, max_deg), dtype=np.int32)
    prob = np.zeros((n, max_deg), dtype=np.float32)
    deg = np.zeros(n, dtype=np.int32)
    for u, v, data in G.edges(data=True):
        idx = deg[u]
        adj[u, idx] = v
        prob[u, idx] = float(data.get("prob", 0.1))
        deg[u] += 1
    return adj, prob

def build_random_limited_graph(G, seed_set, l):
    """
    BFS-style expansion from seed_set. For every visited node, keep up to l random outgoing edges.
    """
    visited = set(seed_set)
    frontier = list(seed_set)
    GD = nx.DiGraph()
    GD.add_nodes_from(G.nodes())

    while frontier:
        next_frontier = []
        for u in frontier:
            nbrs = list(G.successors(u))
            if len(nbrs) == 0:
                continue
            selected = nbrs if len(nbrs) <= l else random.sample(nbrs, l)
            for v in selected:
                if G.has_edge(u, v):
                    GD.add_edge(u, v, prob=G[u][v]["prob"])
                    if v not in visited:
                        next_frontier.append(v)
                        visited.add(v)
        frontier = next_frontier

    return GD

def compute_benefit(activated_mask, benefits_array):
    return float(np.sum(benefits_array[activated_mask]))

def load_costs_benefits(cost_path="cost.txt", benefit_path="benefit.txt"):
    with open(cost_path) as f:
        costs = ast.literal_eval(f.read())
    with open(benefit_path) as f:
        benefits = ast.literal_eval(f.read())
    return {int(k): float(v) for k, v in costs.items()}, {int(k): float(v) for k, v in benefits.items()}

def append_rows_csv(path, rows):
    if not rows: return
    df = pd.DataFrame(rows)
    file_exists = os.path.exists(path)
    df.to_csv(path, mode='a', header=not file_exists, index=False)

def random_seedset_under_budget(nodes, costs, budget):
    """
    Randomly generate a seed set whose total cost <= budget.
    Strategy: shuffle nodes; add nodes while affordable. If nothing fits, return [].
    """
    remaining = float(budget)
    seeds = []
    shuffled = list(nodes)
    random.shuffle(shuffled)
    for v in shuffled:
        c = float(costs.get(int(v), 0.0))
        if c <= remaining:
            seeds.append(v)
            remaining -= c
    return seeds, remaining

# --- Algorithm ---
def run_random_pmcsn_budget(G, budget, l, costs, benefits):
    total_start = time.perf_counter()

    # 1) Random seed set under budget
    seed_set, remaining_budget = random_seedset_under_budget(list(G.nodes()), costs, budget)

    # 2) Build degree-limited graph from seed frontier
    GD = build_random_limited_graph(G, seed_set, l)

    # 3) Matrices + benefits array
    adj_matrix, prob_matrix = to_matrix(GD)
    n = adj_matrix.shape[0]
    benefits_array = np.zeros(n, dtype=np.float32)
    for node, val in benefits.items():
        if 0 <= node < n:
            benefits_array[node] = float(val)

    # 4) Simulate
    sims = simulate_parallel(seed_set, adj_matrix, prob_matrix, sims=SIMULATIONS)
    benefits_list = [compute_benefit(sim[0], benefits_array) for sim in sims]
    steps_list = [sim[1] for sim in sims]
    activations = [sim[2] for sim in sims]

    # 5) Metrics
    avg_benefit = float(np.mean(benefits_list)) if benefits_list else 0.0
    avg_steps = float(np.mean(steps_list)) if steps_list else 0.0
    avg_activated = float(np.mean(activations)) if activations else 0.0
    seed_cost = float(sum(costs.get(int(v), 0.0) for v in seed_set))
    profit = avg_benefit - seed_cost

    return {
        "Model Name": "Random PMCSN (Budget)",
        "Seed Set": seed_set,
        "Seed Set Size": len(seed_set),
        "Seed Set Cost": seed_cost,
        "Remaining Budget After Selection": float(budget) - seed_cost,
        "Benefit": avg_benefit,
        "Profit Earned": profit,
        "Average Timestep": avg_steps,
        "Average Activated Nodes": avg_activated,
        "Value of l": l,
        "Budget": float(budget),
        "Total Nodes": GD.number_of_nodes(),
        "Total Edges": GD.number_of_edges(),
        "Execution Time (s)": float(time.perf_counter() - total_start)
    }

# --- Runner ---
def main():
    if os.path.exists(OUT_CSV):
        os.remove(OUT_CSV)

    G = nx.read_edgelist(GRAPH_FILE, create_using=nx.DiGraph(), nodetype=int, data=(("prob", float),))
    costs, benefits = load_costs_benefits()
    results = []

    for B, l in tqdm([(B, l) for B in BUDGET_LIST for l in L_LIST], desc="(Budget,l) grid"):
        row = run_random_pmcsn_budget(G, B, l, costs, benefits)
        results.append(row)
        append_rows_csv(OUT_CSV, [row])

    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Done.\nCSV: {OUT_CSV}\nJSON: {OUT_JSON}")

if __name__ == "__main__":
    main()

