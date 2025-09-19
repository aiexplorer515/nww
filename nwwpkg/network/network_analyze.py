import plotly.graph_objects as go
import networkx as nx

def draw_network(network_report, events):
    G = nx.Graph()
    for evt in events:
        actors = evt.get("actors", [])
        for a in actors:
            G.add_node(a)
        for i in range(len(actors)):
            for j in range(i+1, len(actors)):
                G.add_edge(actors[i], actors[j])

    pos = nx.spring_layout(G, seed=42)
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    node_x, node_y, text = [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        text.append(node)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y,
                             mode='lines',
                             line=dict(width=0.5, color='#888'),
                             hoverinfo='none'))
    fig.add_trace(go.Scatter(x=node_x, y=node_y,
                             mode='markers+text',
                             text=text,
                             textposition="bottom center",
                             marker=dict(size=15, color='skyblue'),
                             hoverinfo='text'))
    fig.update_layout(showlegend=False,
                      margin=dict(l=0, r=0, t=0, b=0))
    return fig
