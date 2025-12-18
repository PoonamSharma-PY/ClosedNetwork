import os
import ast
import json
import networkx as nx
import numpy as np
from numba import njit
from tqdm import tqdm

# --- Configuration ---
INPUT_FOLDER = "Closed_Network_Outputs_l2_k2_x1000"
SEED_FILE = os.path.join(INPUT_FOLDER, "seed_sets.json")
COST_FILE = "cost.txt"
BENEFIT_FILE = "benefit.txt"
SIMULATIONS = 10000

# --- Load Cost and Benefit Data ---
def load_data():
    with open(COST_FILE) as f:
        costs = ast.literal_eval(f.read())
    with open(BENEFIT_FILE) as f:
        benefits = ast.literal_eval(f.read())
    return {int(k): float(v) for k, v in costs.items()}, {int(k): float(v) for k, v in benefits.items()}

cost_dict, benefit_dict = load_data()

# --- Load Seed Sets ---
with open(SEED_FILE) as f:
    seed_sets = json.load(f)

# --- Numba-accelerated ICM Simulation ---
@njit
def simulate_icm(seed_set, adj_matrix, prob_matrix):
    n = len(adj_matrix)
    activated = np.zeros(n, dtype=np.bool_)
    newly_activated = np.zeros(n, dtype=np.bool_)
    for node in seed_set:
        activated[node] = True
        newly_activated[node] = True
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
    return activated

# --- Compute Benefit ---
def compute_benefit(activated, benefits):
    return sum(benefits.get(i, 0) for i in range(len(activated)) if activated[i])

# --- Convert Graph to Matrix Form ---
def graph_to_matrix(G):
    n = max(G.nodes()) + 1
    max_deg = max(dict(G.out_degree()).values()) if G.number_of_nodes() > 0 else 0
    adj = -np.ones((n, max_deg), dtype=np.int32)
    prob = np.zeros((n, max_deg), dtype=np.float32)
    deg = np.zeros(n, dtype=np.int32)
    for u, v, data in G.edges(data=True):
        idx = deg[u]
        adj[u, idx] = v
        prob[u, idx] = data.get("prob", 0.1)
        deg[u] += 1
    return adj, prob

# --- Evaluate Profit of All Closed Networks ---
best_profit = float("-inf")
best_graph_info = {}

for graph_id, seed_set in tqdm(seed_sets.items(), desc="Evaluating Closed Networks"):
    file_path = os.path.join(INPUT_FOLDER, f"{graph_id}.txt")
    if not os.path.exists(file_path):
        continue

    # Read the graph
    G = nx.DiGraph()
    with open(file_path) as f:
        for line in f:
            u, v, p = line.strip().split()
            G.add_edge(int(u), int(v), prob=float(p))

    adj, prob = graph_to_matrix(G)

    # Run simulations
    benefits_list = []
    for _ in range(SIMULATIONS):
        activated = simulate_icm(np.array(seed_set, dtype=np.int32), adj, prob)
        benefit = compute_benefit(activated, benefit_dict)
        benefits_list.append(benefit)

    avg_benefit = np.mean(benefits_list)
    seed_cost = sum(cost_dict.get(int(n), 0) for n in seed_set)
    profit = avg_benefit - seed_cost

    if profit > best_profit:
        best_profit = profit
        best_graph_info = {
            "Graph_ID": graph_id,
            "Seed_Set": seed_set,
            "Seed_Cost": seed_cost,
            "Avg_Benefit": avg_benefit,
            "Profit": profit,
            "Total_Nodes": G.number_of_nodes(),
            "Total_Edges": G.number_of_edges()
        }

# --- Display Result ---
print("\nBest Closed Network (Maximum Profit):")
for key, value in best_graph_info.items():
    print(f"{key}: {value}")
