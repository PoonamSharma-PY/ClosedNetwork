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

# ---------------- Configuration ----------------
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

NUM_CPUS = 24
SIMULATIONS = 10000          # diffusion simulations for final evaluation
SIMS_PER_CANDIDATE = 100     # sims during candidate selection
EPSILON = 0.1

BUDGET_LIST = [500, 1000, 1500, 2000, 2500]
L_LIST = [4, 8, 12, 16, 20, 24, 28]

GRAPH_FILE = "euemail_trivalency.txt"          # base graph (u v p)
MODEL_NAME = "Trivial PMCSN (Budget)"
OUT_CSV = "Trivial_PMCSN_Bl_grid_results.csv"
OUT_JSON = "Trivial_PMCSN_Bl_grid_results.json"

# Optional: fix RNG for repeatability
# random.seed(12345)
# np.random.seed(12345)

# ---------------- Numba ICM ----------------
@njit
def simulate_icm_numba(seed_set, adj_matrix, prob_matrix):
    """
    Returns:
      activated (bool array),
      steps (rounds executed),
      active_count (int)
    """
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
                    if (not activated[v]) and (np.random.rand() < prob_matrix[u, j]):
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

# ---------------- Helpers ----------------
def to_matrix(graph: nx.DiGraph):
    """Pack DiGraph into (adj, prob) with row-wise variable length using -1 padding."""
    if graph.number_of_nodes() == 0:
        return (np.zeros((0, 0), dtype=np.int32),
                np.zeros((0, 0), dtype=np.float32))
    n = max(graph.nodes()) + 1
    out_deg = dict(graph.out_degree())
    max_deg = max(out_deg.values()) if out_deg else 0
    adj = -np.ones((n, max_deg), dtype=np.int32)
    prob = np.zeros((n, max_deg), dtype=np.float32)
    deg = np.zeros(n, dtype=np.int32)
    for u, v, data in graph.edges(data=True):
        idx = deg[u]
        adj[u, idx] = v
        prob[u, idx] = float(data.get("prob", 0.1))
        deg[u] += 1
    return adj, prob

def build_degree_based_graph(G: nx.DiGraph, l: int) -> nx.DiGraph:
    """Keep for each node only the top-l successors by their out-degree in the original G."""
    GD = nx.DiGraph()
    GD.add_nodes_from(G.nodes())
    for v in G.nodes():
        nbrs = list(G.successors(v))
        nbrs_sorted = sorted(((u, G.out_degree(u)) for u in nbrs),
                             key=lambda x: x[1], reverse=True)
        for u, _deg in nbrs_sorted[:l]:
            if G.has_edge(v, u):
                GD.add_edge(v, u, prob=float(G[v][u].get('prob', 0.1)))
    return GD

def load_costs_benefits(cost_path="cost.txt", benefit_path="benefit.txt"):
    with open(cost_path) as f:
        costs = ast.literal_eval(f.read())
    with open(benefit_path) as f:
        benefits = ast.literal_eval(f.read())
    costs = {int(k): float(v) for k, v in costs.items()}
    benefits = {int(k): float(v) for k, v in benefits.items()}
    return costs, benefits

def compute_benefit_from_mask(mask: np.ndarray, benefits_array: np.ndarray) -> float:
    return float(np.sum(benefits_array[mask]))

def append_rows_csv(path, rows):
    """Append list[dict] to CSV, create with header if missing."""
    if not rows:
        return
    df = pd.DataFrame(rows)
    file_exists = os.path.exists(path)
    df.to_csv(path, mode='a', header=not file_exists, index=False)

# ---------------- Core Algorithm (one Budget, l) ----------------
def trivial_pmcsn_for_B_l(G: nx.DiGraph, budget: float, l: int, costs: dict, benefits: dict, model_name: str, graph_id: str):
    """
    Greedy selection under a total budget:
      - Build degree-limited graph by l.
      - Iteratively add a node v (cost<=remaining) that maximizes marginal benefit per cost,
        using SIMS_PER_CANDIDATE sims for estimation, until budget is exhausted or no feasible candidate remains.
      - Evaluate final S with SIMULATIONS runs.
    """
    total_start = time.perf_counter()

    # Degree-limited graph for this l
    GD = build_degree_based_graph(G, l)
    total_nodes = GD.number_of_nodes()
    total_edges = GD.number_of_edges()

    # Matrix + benefits array aligned to node IDs
    adj_matrix, prob_matrix = to_matrix(GD)
    n = adj_matrix.shape[0]
    benefits_array = np.zeros(n, dtype=np.float32)
    for node_id, val in benefits.items():
        if 0 <= node_id < n:
            benefits_array[node_id] = float(val)

    # Greedy seed selection under budget
    S = []
    spent = 0.0
    remaining_budget = float(budget)

    # candidate sampling size (same spirit as original; independent of budget)
    candidate_sample_size = max(1, int((max(1, total_nodes) / max(1, 10)) * math.log(1.0 / EPSILON)))
    # note: the divisor "10" above is arbitrary to keep sample size reasonable without k.
    # You can tune it if you want a larger/smaller candidate pool per iteration.

    # Precompute base benefit of S=∅ once to avoid re-sim each loop start
    base_benef = 0.0
    base_sims_cached = None  # when S changes, recompute

    while True:
        # Feasible remaining nodes by cost
        remaining_nodes = [v for v in GD.nodes() if (v not in S) and (costs.get(int(v), 0.0) <= remaining_budget)]
        if not remaining_nodes:
            break

        sample_size = min(candidate_sample_size, len(remaining_nodes))
        candidates = random.sample(remaining_nodes, sample_size)

        # compute base benefit for current S if not cached
        if S and (base_sims_cached is None):
            base_sims = simulate_parallel(S, adj_matrix, prob_matrix, sims=SIMS_PER_CANDIDATE)
            base_benef = float(np.mean([compute_benefit_from_mask(sim[0], benefits_array) for sim in base_sims]))
            base_sims_cached = True
        elif not S:
            base_benef = 0.0
            base_sims_cached = True

        best_score = -np.inf
        best_node = None
        best_gain = 0.0
        best_cost = 0.0

        for v in candidates:
            c_v = float(costs.get(int(v), 0.0))
            if c_v <= 0.0 or c_v > remaining_budget:
                continue
            new_S = S + [v]
            new_sims = simulate_parallel(new_S, adj_matrix, prob_matrix, sims=SIMS_PER_CANDIDATE)
            new_benef = float(np.mean([compute_benefit_from_mask(sim[0], benefits_array) for sim in new_sims]))
            gain = new_benef - base_benef
            score = gain / c_v  # marginal benefit per cost
            if score > best_score:
                best_score = score
                best_node = v
                best_gain = gain
                best_cost = c_v

        if best_node is None:
            # no feasible improving candidate under remaining budget
            break

        # accept best_node
        S.append(best_node)
        spent += best_cost
        remaining_budget = float(budget) - spent
        # S changed => invalidate cached base sims so we recompute in next loop
        base_sims_cached = None

    # Final evaluation on S
    final_sims = simulate_parallel(S, adj_matrix, prob_matrix, sims=SIMULATIONS)
    avg_benefit = float(np.mean([compute_benefit_from_mask(sim[0], benefits_array) for sim in final_sims]))
    avg_timestep = float(np.mean([sim[1] for sim in final_sims]))
    avg_activated = float(np.mean([sim[2] for sim in final_sims]))

    seed_cost = float(sum(costs.get(int(v), 0.0) for v in S))
    profit = avg_benefit - seed_cost
    remaining_budget_after = float(budget) - seed_cost

    total_time = float(time.perf_counter() - total_start)

    return {
        "Model Name": model_name,
        "Seed Set": S,
        "Seed Set Size": int(len(S)),
        "Seed Set Cost": seed_cost,
        "Remaining Budget After Selection": remaining_budget_after,
        "Benefit": avg_benefit,
        "Profit Earned": profit,
        "Average Timestep": avg_timestep,
        "Average Activated Nodes": avg_activated,
        "Value of l": int(l),
        "Budget": float(budget),
        "Graph Id": graph_id,          # base graph file (or tag)
        "Total Nodes": int(total_nodes),
        "Total Edges": int(total_edges),
        "Total-(B,l) Execution Time (s)": total_time
    }

# ---------------- Runner over all (Budget, l) ----------------
def main():
    # Fresh outputs (comment these 3 lines if you want to append/resume)
    for p in (OUT_CSV, OUT_JSON):
        if os.path.exists(p):
            os.remove(p)

    print("Reading graph and input data...")
    G = nx.read_edgelist(GRAPH_FILE, create_using=nx.DiGraph(), nodetype=int, data=(("prob", float),))
    costs, benefits = load_costs_benefits()

    results = []
    pbar = tqdm([(B, l) for B in BUDGET_LIST for l in L_LIST], desc="(Budget,l) combos")

    for B, l in pbar:
        pbar.set_postfix({"Budget": B, "l": l})
        row = trivial_pmcsn_for_B_l(G, B, l, costs, benefits, MODEL_NAME, os.path.basename(GRAPH_FILE))
        results.append(row)
        append_rows_csv(OUT_CSV, [row])  # stream to CSV

    # Save JSON too (optional)
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)

    # Quick print: top 5 by Profit
    df = pd.DataFrame(results)
    df_sorted = df.sort_values("Profit Earned", ascending=False).head(5)
    print("\nTop 5 (by Profit Earned):")
    for _, r in df_sorted.iterrows():
        print(f"(B={r['Budget']}, l={r['Value of l']}): Profit={r['Profit Earned']:.6f}, SeedSize={r['Seed Set Size']}")

    print(f"\nGrid run complete.\nCSV: {OUT_CSV}\nJSON: {OUT_JSON}")

if __name__ == "__main__":
    main()
