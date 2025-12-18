# highdeg_pmcsn_budget.py
import os
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
OUT_CSV = "HighDeg_PMCSN_Bl.csv"
OUT_JSON = "HighDeg_PMCSN_Bl.json"

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
    """
    Convert DiGraph to fixed-size adjacency/probability matrices for numba.
    Rows indexed by node id (assumes integer node labels). Columns up to max out-degree.
    Unused entries are -1 in adj and 0 in prob.
    """
    if G.number_of_nodes() == 0:
        return np.empty((0, 0), dtype=np.int32), np.empty((0, 0), dtype=np.float32)

    n = max(G.nodes()) + 1
    out_deg = dict(G.out_degree())
    max_deg = max(out_deg.values(), default=0)

    if max_deg == 0:
        # no edges at all
        return -np.ones((n, 1), dtype=np.int32), np.zeros((n, 1), dtype=np.float32)

    adj = -np.ones((n, max_deg), dtype=np.int32)
    prob = np.zeros((n, max_deg), dtype=np.float32)
    fill = np.zeros(n, dtype=np.int32)

    for u, v, data in G.edges(data=True):
        if u < 0 or v < 0:
            continue
        idx = fill[u]
        if idx < max_deg:
            adj[u, idx] = v
            prob[u, idx] = float(data.get("prob", 0.1))
            fill[u] += 1

    return adj, prob

def build_random_limited_graph(G, seed_set, l):
    """
    Starting from seed_set, expand a frontier. For each visited node, keep up to l random
    outgoing edges. Add those edges to GD and continue from reached neighbors.
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
                        visited.add(v)
                        next_frontier.append(v)
        frontier = next_frontier

    return GD

def compute_benefit(activated_mask, benefits_array):
    return float(np.sum(benefits_array[activated_mask]))

def load_costs_benefits(cost_path="cost.txt", benefit_path="benefit.txt"):
    with open(cost_path, "r") as f:
        costs = ast.literal_eval(f.read())
    with open(benefit_path, "r") as f:
        benefits = ast.literal_eval(f.read())
    # normalize keys->int, values->float
    return {int(k): float(v) for k, v in costs.items()}, {int(k): float(v) for k, v in benefits.items()}

def append_rows_csv(path, rows):
    if not rows:
        return
    df = pd.DataFrame(rows)
    file_exists = os.path.exists(path)
    df.to_csv(path, mode="a", header=not file_exists, index=False)

def select_topdeg_under_budget(G, costs, budget):
    """
    Greedily select nodes by descending out-degree, adding while total cost <= budget.
    Tie-breaker: node id ascending.
    Returns (seed_set, remaining_budget).
    """
    od = list(G.out_degree())
    od.sort(key=lambda x: (-x[1], x[0]))  # high out-degree first, then id
    selected = []
    spent = 0.0
    for u, _d in od:
        c = float(costs.get(int(u), 0.0))
        if spent + c <= budget:
            selected.append(u)
            spent += c
    return selected, float(budget - spent)

# --- Algorithm ---
def run_highdeg_pmcsn_budget(G, budget, l, costs, benefits):
    start = time.perf_counter()

    # 1) Seeds = highest out-degree nodes that fit within budget
    seed_set, remaining_after = select_topdeg_under_budget(G, costs, float(budget))

    # 2) Build degree-limited graph outward from seeds
    GD = build_random_limited_graph(G, seed_set, l)

    # 3) Convert to matrices for numba simulation
    adj_matrix, prob_matrix = to_matrix(GD)
    n = adj_matrix.shape[0]

    # 4) Align benefits to matrix index space
    benefits_array = np.zeros(n, dtype=np.float32)
    for node, val in benefits.items():
        if 0 <= node < n:
            benefits_array[node] = float(val)

    # 5) Simulate ICM
    sims = simulate_parallel(seed_set, adj_matrix, prob_matrix, sims=SIMULATIONS)
    benefits_list = [compute_benefit(sim[0], benefits_array) for sim in sims]
    steps_list = [sim[1] for sim in sims]
    activations = [sim[2] for sim in sims]

    # 6) Aggregate & profit
    avg_benefit = float(np.mean(benefits_list)) if benefits_list else 0.0
    avg_steps = float(np.mean(steps_list)) if steps_list else 0.0
    avg_activated = float(np.mean(activations)) if activations else 0.0
    seed_cost = float(sum(costs.get(v, 0.0) for v in seed_set))
    profit = avg_benefit - seed_cost

    return {
        "Model Name": "HighDeg PMCSN (Budget)",
        "Seed Set": seed_set,
        "Seed Set Size": len(seed_set),
        "Seed Set Cost": seed_cost,
        "Remaining Budget After Selection": remaining_after,
        "Benefit": avg_benefit,
        "Profit Earned": profit,
        "Average Timestep": avg_steps,
        "Average Activated Nodes": avg_activated,
        "Value of l": l,
        "Budget": float(budget),
        "Total Nodes": GD.number_of_nodes(),
        "Total Edges": GD.number_of_edges(),
        "Execution Time (s)": float(time.perf_counter() - start),
    }

# --- Runner ---
def main():
    # Fresh CSV each run
    if os.path.exists(OUT_CSV):
        os.remove(OUT_CSV)

    # Graph: "u v prob"
    G = nx.read_edgelist(
        GRAPH_FILE,
        create_using=nx.DiGraph(),
        nodetype=int,
        data=(("prob", float),),
    )

    costs, benefits = load_costs_benefits()
    results = []

    for B, l in tqdm([(B, l) for B in BUDGET_LIST for l in L_LIST], desc="(Budget,l) grid"):
        row = run_highdeg_pmcsn_budget(G, B, l, costs, benefits)
        results.append(row)
        append_rows_csv(OUT_CSV, [row])

    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n Done.\nCSV: {OUT_CSV}\nJSON: {OUT_JSON}")

if __name__ == "__main__":
    main()
