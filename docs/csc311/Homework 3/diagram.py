import matplotlib.pyplot as plt
import networkx as nx

# Create a directed graph
G = nx.DiGraph()

# Define the nodes and edges
G.add_node("p(θ)")
G.add_node("p(y)")
G.add_node("p(x|y,θ)")
G.add_node("p(x,y|θ)")
G.add_node("p(x|θ)")
G.add_node("p(y|x,θ)")

# Add edges to show relationships
G.add_edge("p(θ)", "p(y|x,θ)")
G.add_edge("p(y)", "p(x,y|θ)")
G.add_edge("p(x|y,θ)", "p(x,y|θ)")
G.add_edge("p(x,y|θ)", "p(x|θ)")
G.add_edge("p(x,y|θ)", "p(y|x,θ)")
G.add_edge("p(x|θ)", "p(y|x,θ)")

# Get positions using a hierarchy layout
pos = {
    "p(θ)": (0, 2),
    "p(y)": (-2, 1),
    "p(x|y,θ)": (2, 1),
    "p(x,y|θ)": (0, 0),
    "p(x|θ)": (-1, -1),
    "p(y|x,θ)": (1, -1)
}

# Draw the graph
plt.figure(figsize=(10, 6))
nx.draw(
    G,
    pos,
    with_labels=True,
    labels={
        "p(θ)": "Prior\np(θ)",
        "p(y)": "Prior\np(y)",
        "p(x|y,θ)": "Likelihood\np(x|y,θ)",
        "p(x,y|θ)": "Joint\np(x,y|θ)",
        "p(x|θ)": "Marginal Likelihood\np(x|θ)",
        "p(y|x,θ)": "Posterior\np(y|x,θ)"
    },
    node_size=5000,
    node_color="lightblue",
    arrows=True,
    arrowstyle="->",
    arrowsize=20,
    font_size=10
)

plt.title("Diagram of Probabilistic Relationships", fontsize=14)
plt.axis("off")
plt.tight_layout()
plt.show()
