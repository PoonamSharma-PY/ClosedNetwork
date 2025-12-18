import os
import random
import networkx as nx
import json


# ----------------------------
# SECTION 1: Read graph
# ----------------------------
def read_graph_with_probabilities(filepath):
    G = nx.DiGraph()
    with open(filepath, 'r') as f:
        for line in f:
            u, v, p = line.strip().split()
            G.add_edge(int(u), int(v), prob=float(p))
    return G

# ----------------------------
# Load costs (Python dict literal: {node: cost, ...})
# ----------------------------
def load_costs(cost_path="cost.txt"):
    with open(cost_path, "r") as f:
        data = f.read().strip()
    # supports Python dict literal or JSON dict
    try:
        costs = json.loads(data)
    except json.JSONDecodeError:
        import ast
        costs = ast.literal_eval(data)
    # normalize keys/values
    return {int(k): float(v) for k, v in costs.items()}

# ----------------------------
# NEW: memory-light seed generator under budget
# ----------------------------
def generate_unique_random_seed_sets_under_budget(nodes, costs, budget, n, rng=None):
    """
    Generate n unique seed sets (sorted tuples) such that sum(costs[seed]) <= budget.
    Memory-light: uses randomized greedy packing; no combinatorial blow-up.

    Strategy:
      - Filter nodes affordable individually (cost <= budget)
      - Repeatedly shuffle affordable nodes and greedily add while keeping total <= budget
      - Ensure uniqueness via set of tuples
    """
    if rng is None:
        rng = random

    affordable = [u for u in nodes if costs.get(u, 0.0) <= budget]
    if not affordable:
        raise ValueError(f"No nodes affordable within budget={budget}. Check costs.")

    seeds = []
    seen = set()
    attempts = 0
    max_attempts = 40 * n  # generous cap

    while len(seeds) < n and attempts < max_attempts:
        attempts += 1
        rng.shuffle(affordable)
        total = 0.0
        chosen = []
        for u in affordable:
            c = costs.get(u, 0.0)
            if total + c <= budget:
                chosen.append(u)
                total += c
        if not chosen:
            # Shouldn't happen because affordable non-empty, but guard anyway
            continue

        ss = tuple(sorted(chosen))
        if ss not in seen:
            seen.add(ss)
            seeds.append(ss)

    if len(seeds) < n:
        raise ValueError(
            f"Could not find {n} unique seed sets within budget={budget}. "
            f"Generated {len(seeds)} after {attempts} attempts. "
            "Try lowering n or increasing budget."
        )
    return seeds

# ----------------------------
# EDGE CHOICES per node (per network)
# ----------------------------
def sample_out_edges_per_node(G, l, rng=None):
    """
    For each node, pick up to l outgoing neighbors.
    Returns a dict: node -> tuple(selected_neighbors)
    """
    if rng is None:
        rng = random
    choices = {}
    for u in G.nodes():
        nbrs = list(G.successors(u))
        if len(nbrs) <= l:
            choices[u] = tuple(nbrs)
        else:
            # sorted for a stable hash
            choices[u] = tuple(sorted(rng.sample(nbrs, l)))
    return choices

# ----------------------------
# STREAMING SAVE (no big lists in memory)
# ----------------------------
def save_closed_network_stream(G, seed_sets, l, n, output_folder, rng=None):
    """
    For a fixed l and a given list of seed_sets (len == n),
    generate n closed networks, write each to disk immediately, and do not keep graphs in RAM.
    """
    if rng is None:
        rng = random
    os.makedirs(output_folder, exist_ok=True)

    seed_map = {}
    seen_configs = set()

    i = 0
    while i < n:
        seed_set = seed_sets[i]  # reuse seeds in given order
        edge_choices = sample_out_edges_per_node(G, l, rng=rng)

        # Hash to ensure uniqueness in this run
        edge_signature = tuple((u, edge_choices[u]) for u in sorted(edge_choices.keys()))
        cfg_hash = (seed_set, edge_signature)
        if cfg_hash in seen_configs:
            continue
        seen_configs.add(cfg_hash)

        graph_id = f"ClosedNetwork_{i+1}"
        path = os.path.join(output_folder, f"{graph_id}.txt")
        with open(path, "w") as f:
            for u, targets in edge_choices.items():
                for v in targets:
                    if G.has_edge(u, v):
                        p = G[u][v].get("prob", 1.0)
                        f.write(f"{u} {v} {p}\n")

        seed_map[graph_id] = list(seed_set)
        i += 1

    with open(os.path.join(output_folder, "seed_sets.json"), "w") as f:
        json.dump(seed_map, f, indent=2)

    print(f"Saved {n} closed networks to {output_folder}")

# ----------------------------
# MAIN
# ----------------------------
def main():
    # Replace k-list with budgets
    BUDGET_LIST = [500, 1000, 1500, 2000, 2500]
    L_LIST = [4, 8, 12, 16, 20, 24, 28]

    x = 1000  # networks per (budget, l)
    input_graph_path = "euemail_trivalency.txt"
    cost_path = "cost.txt"  # {node: cost, ...}

    # Reproducibility (optional)
    random.seed(12345)

    G = read_graph_with_probabilities(input_graph_path)
    nodes = list(G.nodes())
    costs = load_costs(cost_path)

    # Save per-budget seed sets (once) and reuse across all l
    seeds_root = "Closed_Network_Seeds"
    os.makedirs(seeds_root, exist_ok=True)

    for budget in BUDGET_LIST:
        # 1) Generate seeds once (memory-light, budget-feasible, random unique)
        seed_sets = generate_unique_random_seed_sets_under_budget(
            nodes, costs, budget, n=x, rng=random
        )

        # Save the per-budget seeds as a separate file (for audit/reuse)
        seeds_path = os.path.join(seeds_root, f"seed_sets_B{budget}_x{x}.json")
        with open(seeds_path, "w") as f:
            json.dump([list(s) for s in seed_sets], f, indent=2)
        print(f"Saved seed sets for budget={budget}: {seeds_path}")

        # 2) For each l, stream-generate and save networks using the SAME seeds
        for l in L_LIST:
            out_dir = f"Closed_Network_Outputs_l{l}_B{budget}_x{x}"
            save_closed_network_stream(G, seed_sets, l, x, out_dir, rng=random)

if __name__ == "__main__":
    main()
