import pandas as pd
import networkx as nx
import plotly.graph_objs as go
import numpy as np
import random as rand
import webbrowser
from sklearn.cluster import SpectralClustering

# --- Data Loader ---
def load_common_followers(file_path):
    df = pd.read_excel(file_path, header=None)
    return df

# --- Graph Construction ---
def build_social_graph(df):
    G = nx.Graph()
    top_users = df.iloc[0].dropna().tolist()

    for col in df.columns:
        main_user = df.iloc[0, col]
        if pd.isna(main_user):
            continue
        for follower in df.iloc[1:, col].dropna():
            G.add_edge(main_user, follower)

    print(f"\nGraph Info:\nNodes: {G.number_of_nodes()} | Edges: {G.number_of_edges()}")
    return G

# --- 3D Visualization ---
def visualize_social_graph_3d(G, clusters=6, scale=2.0, show_labels=True, filename="social_graph_3d.html"):
    nodes = list(G.nodes())
    
    # Adjacency matrix for clustering
    adj_matrix = nx.to_numpy_array(G, nodelist=nodes)
    sc = SpectralClustering(
        n_clusters=clusters,
        affinity='precomputed',
        random_state=rand.randint(1, 1000)
    )
    labels = sc.fit_predict(adj_matrix)

    # Get 3D positions
    pos = nx.spring_layout(
        G, 
        dim=3, 
        k=1.0,
        iterations=10000,
        weight=None,
        seed=rand.randint(1, 1000)
    )
    
    # Scale layout
    for node in pos:
        pos[node] = tuple(coord * scale for coord in pos[node])

    xs = [pos[node][0] for node in nodes]
    ys = [pos[node][1] for node in nodes]
    zs = [pos[node][2] for node in nodes]

    # Plotly figure
    fig = go.Figure()

    # Nodes
    fig.add_trace(go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode='markers+text' if show_labels else 'markers',
        marker=dict(
            size=12,
            color=labels,
            colorscale='Viridis',
            opacity=0.9,
            line=dict(color='black', width=1.5)
        ),
        text=nodes if show_labels else None,
        textfont=dict(size=12, color='white'),
        hoverinfo='text',
        name='Users'
    ))

    # Edges
    for u, v in G.edges():
        fig.add_trace(go.Scatter3d(
            x=[pos[u][0], pos[v][0]],
            y=[pos[u][1], pos[v][1]],
            z=[pos[u][2], pos[v][2]],
            mode='lines',
            line=dict(color='rgba(150,150,150,0.4)', width=1),
            hoverinfo='none',
            showlegend=False
        ))

    # Layout
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False)
        ),
        paper_bgcolor='black',
        margin=dict(l=0, r=0, b=0, t=40)
    )

    # Save and open
    fig.write_html(filename)
    webbrowser.open(filename)
    print(f"Graph saved as {filename}")

# --- Main ---
if __name__ == "__main__":
    # 1. Load Excel
    df = load_common_followers("common.xlsx")

    # 2. Build graph
    G = build_social_graph(df)

    # 3. Visualize (adjust options as needed)
    visualize_social_graph_3d(G, clusters=3, scale=2, show_labels=True)
    
