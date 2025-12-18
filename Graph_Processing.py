import networkx as nx
import random

# Configuration
INPUT_FILES = {
    'graph': "euemail.txt"
    # 'cost': "cost.txt", 
    # 'benefit': "benefit.txt"
}

# Output files for the different graph versions
OUTPUT_FILES = {
    'uniform': "euemail_uniform.txt",
    'trivalency': "euemail_trivalency.txt",
    'weighted': "euemail_weighted.txt"
}

def load_data():
    """Load graph and attribute files"""
    # Read the graph (will be either directed or undirected based on input)
    G = nx.read_edgelist(INPUT_FILES['graph'], create_using=nx.DiGraph(), nodetype=int)
    # G = nx.read_edgelist(INPUT_FILES['graph'], create_using=nx.Graph(), nodetype=int)
    print(f"Loaded as {'directed' if nx.is_directed(G) else 'undirected'} graph")
    return G

def create_graph_versions(G):
    """Create 3 versions of the graph with different probability assignments"""
    # Version 1: Uniform 0.1 probability
    uniform_G = G.copy()
    for u, v in uniform_G.edges():
        uniform_G[u][v]['weight'] = 0.1

    # Version 2: Trivalency (random choice from [0.1, 0.01, 0.001])
    trivalency_G = G.copy()
    trivalency_values = [0.1, 0.01, 0.001]
    for u, v in trivalency_G.edges():
        trivalency_G[u][v]['weight'] = random.choice(trivalency_values)

    # Version 3: Weighted probability (direction-aware)
    weighted_G = G.copy()

    if nx.is_directed(weighted_G):
        # Directed graph - use in-degree normalization
        print("Applying directed weighting (1/in_degree)")
        for u, v in weighted_G.edges():
            in_degree = weighted_G.in_degree(v)
            weighted_G[u][v]['weight'] = round(1.0 / in_degree, 3) if in_degree > 0 else 0.0
    else:
        # Undirected graph - use degree normalization
        print("Applying undirected weighting (1/degree)")
        for u, v in weighted_G.edges():
            degree = weighted_G.degree(v)
            weighted_G[u][v]['weight'] = round(1.0 / degree, 3) if degree > 0 else 0.0

    return uniform_G, trivalency_G, weighted_G


def save_graph(G, filename):
    """Save graph in edge list format with weights"""
    with open(filename, 'w') as f:
        for u, v, data in G.edges(data=True):
            weight = data.get('weight', 1.0)  # default to 1.0 if no weight
            f.write(f"{u} {v} {weight}\n")

def load_weighted_graph(filename):
    """Load a graph with edge weights"""
    G = nx.read_weighted_edgelist(filename, create_using=nx.DiGraph(), nodetype=int)
    return G

def main():
    # Load original graph
    G = load_data()
    
    # Create different versions
    uniform_G, trivalency_G, weighted_G = create_graph_versions(G)
    
    # Save the versions
    save_graph(uniform_G, OUTPUT_FILES['uniform'])
    save_graph(trivalency_G, OUTPUT_FILES['trivalency'])
    save_graph(weighted_G, OUTPUT_FILES['weighted'])
    
    print("Graph versions created and saved successfully!")
    
    # Example of how to load them back
    loaded_weighted = load_weighted_graph(OUTPUT_FILES['weighted'])
    print(f"Loaded weighted graph as {'directed' if nx.is_directed(loaded_weighted) else 'undirected'}")
    
    # Show sample edge
    sample_edge = next(iter(loaded_weighted.edges(data=True)), None)
    if sample_edge:
        u, v, data = sample_edge
        print(f"Sample edge: {u} -> {v} with weight {data.get('weight', 'N/A')}")
    else:
        print("Graph has no edges")

if __name__ == "__main__":
    main()
