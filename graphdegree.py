import networkx as nx

# Example: Load a graph from an edge list file (format: u v p)
G = nx.DiGraph()
with open("Closed_Network_Outputs_l2_k2_x1000/ClosedNetwork_3.txt") as f:
    for line in f:
        u, v, _ = line.strip().split()
        G.add_edge(int(u), int(v))

# Calculate degrees
for node in G.nodes():
    in_deg = G.in_degree(node)
    out_deg = G.out_degree(node)
    total_deg = in_deg + out_deg
    print(f"Node {node}: In-Degree = {in_deg}, Out-Degree = {out_deg}, Total Degree = {total_deg}")
