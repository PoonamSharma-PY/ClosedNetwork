import os
import re
import ast
import json
import time
import glob
import networkx as nx
import numpy as np
import pandas as pd
from numba import njit
from tqdm import tqdm

# ---------------- Configuration ----------------
ROOT_DIR = "."
# UPDATED: folder pattern now uses B (budget) instead of k
FOLDER_GLOB = os.path.join(ROOT_DIR, "Closed_Network_Outputs_l*_B*_x*")

COST_FILE = "cost.txt"
BENEFIT_FILE = "benefit.txt"
SIMULATIONS = 10000  # Number of diffusion simulations per closed network (per graph)

# UPDATED: output names reference Budget (B) and l
BEST_PER_BL_JSON = "best_closed_network_per_B_l.json"
BEST_PER_BL_CSV  = "best_closed_network_per_B_l.csv"
PROGRESS_ALL_GRAPHS_CSV = "progress_all_graphs.csv"

# Helpful label for "Model Name" (purely cosmetic)
INPUT_GRAPH_PATH = "euemail_trivalency.txt"  # used only for naming in output


# ---------------- Utilities ----------------
def load_cost_benefit(cost_path, benefit_path):
    with open(cost_path) as f:
        costs = ast.literal_eval(f.read())
    with open(benefit_path) as f:
        benefits = ast.literal_eval(f.read())
    return {int(k): float(v) for k, v in costs.items()}, {int(k): float(v) for k, v in benefits.items()}

def parse_folder_bl(folder_name):
    """
    Parses l, B, x from folder names like: Closed_Network_Outputs_l8_B1000_x2000
    Returns (l_val, budget_val, x_val)
    """
    m = re.search(r"_l(\d+)_B(\d+)_x(\d+)", folder_name)
    if not m:
        raise ValueError(f"Cannot parse B/l/x from folder name: {folder_name}")
    return int(m.group(1)), int(m.group(2)), int(m.group(3))

def graph_to_matrix(G):
    n = max(G.nodes()) + 1 if G.number_of_nodes() > 0 else 0
    if n == 0:
        return np.zeros((0, 0), dtype=np.int32), np.zeros((0, 0), dtype=np.float32)
    out_deg = dict(G.out_degree())
    max_deg = max(out_deg.values()) if out_deg else 0
    adj = -np.ones((n, max_deg), dtype=np.int32)
    prob = np.zeros((n, max_deg), dtype=np.float32)
    deg = np.zeros(n, dtype=np.int32)
    for u, v, data in G.edges(data=True):
        idx = deg[u]
        adj[u, idx] = v
        prob[u, idx] = float(data.get("prob", 0.1))
        deg[u] += 1
    return adj, prob

def append_rows_csv(path, rows):
    """Append list[dict] to CSV, create with header if missing."""
    if not rows:
        return
    df = pd.DataFrame(rows)
    file_exists = os.path.exists(path)
    df.to_csv(path, mode='a', header=not file_exists, index=False)


# ---------------- Numba ICM ----------------
@njit
def simulate_icm(seed_set, adj_matrix, prob_matrix):
    """
    Returns:
      activated (bool array),
      steps (rounds executed)
    """
    n = adj_matrix.shape[0]
    activated = np.zeros(n, dtype=np.bool_)
    newly_activated = np.zeros(n, dtype=np.bool_)
    for node in seed_set:
        if node < n:
            activated[node] = True
            newly_activated[node] = True
    steps = 0
    if n == 0:
        return activated, steps
    while True:
        any_new = False
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
                        any_new = True
        if not any_new:
            break
        newly_activated = next_new
        steps += 1
    return activated, steps

def benefit_from_mask(mask, benefits_dict):
    total = 0.0
    n = len(mask)
    for i in range(n):
        if mask[i]:
            total += benefits_dict.get(i, 0.0)
    return total


# ---------------- Main ----------------
def main():
    model_name = os.path.splitext(os.path.basename(INPUT_GRAPH_PATH))[0]
    cost_dict, benefit_dict = load_cost_benefit(COST_FILE, BENEFIT_FILE)

    # Discover folders (Budget-based)
    folders = sorted(glob.glob(FOLDER_GLOB))
    if not folders:
        raise FileNotFoundError(f"No folders found with pattern: {FOLDER_GLOB}")

    # (Optional) Fresh run: clear outputs. Comment out if you want to resume.
    for p in (BEST_PER_BL_CSV, PROGRESS_ALL_GRAPHS_CSV, BEST_PER_BL_JSON):
        if os.path.exists(p):
            os.remove(p)

    winners = []           # one best row per (B,l)
    global_best = None     # best across all (B,l)

    # Process each (B,l) folder
    for folder in tqdm(folders, desc="Folders"):
        folder_name = os.path.basename(folder.rstrip("/\\"))
        try:
            l_val, budget_val, x_val = parse_folder_bl(folder_name)  # x_val == No. of Samples per (B,l)
        except ValueError:
            continue

        seed_file = os.path.join(folder, "seed_sets.json")
        if not os.path.exists(seed_file):
            continue

        with open(seed_file, "r") as f:
            seed_map = json.load(f)  # {Graph_ID: [seed nodes...]}

        best_row = None  # winner for this (B,l)
        progress_buffer = []  # buffer to append progress in chunks

        # --- Measure total time to find best for this (B,l) ---
        folder_t0 = time.perf_counter()

        # Iterate all graphs in this folder (~x_val samples)
        for graph_id, seed_set in tqdm(seed_map.items(), desc=f"{folder_name}", leave=False):
            graph_path = os.path.join(folder, f"{graph_id}.txt")
            if not os.path.exists(graph_path):
                continue

            # Load graph
            G = nx.DiGraph()
            with open(graph_path, "r") as f:
                for line in f:
                    u, v, p = line.strip().split()
                    G.add_edge(int(u), int(v), prob=float(p))

            total_nodes = G.number_of_nodes()
            total_edges = G.number_of_edges()
            adj, prob = graph_to_matrix(G)
            seed_arr = np.array([int(x) for x in seed_set], dtype=np.int32)

            sim_benefits = np.empty(SIMULATIONS, dtype=np.float64)
            sim_steps    = np.empty(SIMULATIONS, dtype=np.int32)
            sim_actives  = np.empty(SIMULATIONS, dtype=np.int32)

            # Per-graph timing (useful for progress CSV)
            g_t0 = time.perf_counter()
            for s in range(SIMULATIONS):
                activated_mask, steps = simulate_icm(seed_arr, adj, prob)
                sim_benefits[s] = benefit_from_mask(activated_mask, benefit_dict)
                sim_steps[s]    = steps
                sim_actives[s]  = int(activated_mask.sum())
            g_exec_time = time.perf_counter() - g_t0

            avg_benefit   = float(sim_benefits.mean())
            avg_timestep  = float(sim_steps.mean())
            avg_activated = float(sim_actives.mean())
            seed_cost     = float(sum(cost_dict.get(int(n), 0.0) for n in seed_set))
            profit        = avg_benefit - seed_cost

            row = {
                "Model Name": model_name,
                "Seed Set": list(map(int, seed_set)),
                "Seed Set Size": int(len(seed_set)),
                "Seed Set Cost": seed_cost,
                "Benefit": avg_benefit,
                "Profit Earned": profit,
                "Average Timestep": avg_timestep,
                "Average Activated Nodes": avg_activated,
                "No. of Samples": int(x_val),           # 🔹 samples per (B,l)
                "Value of l": int(l_val),
                "Budget": int(budget_val),              # 🔹 replaces k
                "Graph Id": graph_id,
                "Total Nodes": int(total_nodes),
                "Total Edges": int(total_edges),
                "Per-Graph Time (s)": g_exec_time       # 🔹 for progress view
            }

            # --- Stream per-graph progress (append) ---
            progress_buffer.append(row)
            if len(progress_buffer) >= 25:  # flush in chunks to reduce I/O
                append_rows_csv(PROGRESS_ALL_GRAPHS_CSV, progress_buffer)
                progress_buffer = []

            # Track best for this (B,l)
            if (best_row is None) or (row["Profit Earned"] > best_row["Profit Earned"]):
                best_row = row

            # Track global best
            if (global_best is None) or (row["Profit Earned"] > global_best["Profit Earned"]):
                global_best = row

        # flush any remaining per-graph progress
        if progress_buffer:
            append_rows_csv(PROGRESS_ALL_GRAPHS_CSV, progress_buffer)

        # Total time to find the best for this (B,l)
        folder_exec_time = time.perf_counter() - folder_t0

        # Save/update best-per-(B,l) CSV immediately after finishing folder
        if best_row is not None:
            # Overwrite the timing field to be the TOTAL time for the (B,l) search
            best_row_out = dict(best_row)
            best_row_out["Execution Time (s)"] = folder_exec_time  # 🔹 total time for best selection over x_val samples

            # Keep the winner list and stream to CSV
            winners.append(best_row_out)
            append_rows_csv(BEST_PER_BL_CSV, [best_row_out])

            print(f"\n🏁 Winner for (B={best_row_out['Budget']}, l={best_row_out['Value of l']}): "
                  f"Graph {best_row_out['Graph Id']} | Profit={best_row_out['Profit Earned']:.6f} | "
                  f"Total Time={best_row_out['Execution Time (s)']:.2f}s")

    # Save final winners JSON
    with open(BEST_PER_BL_JSON, "w") as f:
        json.dump(winners, f, indent=2)

    # Print global best (by profit)
    if global_best:
        print("\n Global Best Across All (B,l):")
        for k, v in global_best.items():
            print(f"{k}: {v}")
        print(f"\n Best-per-(B,l): {BEST_PER_BL_CSV}")
        print(f" Progress (all graphs): {PROGRESS_ALL_GRAPHS_CSV}")
    else:
        print("No results recorded. Check folder pattern and seed files.")

if __name__ == "__main__":
    main()
