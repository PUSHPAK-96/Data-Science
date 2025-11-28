import pandas as pd
import networkx as nx
import plotly.graph_objects as go


def top_n_products(df: pd.DataFrame, n: int = 10) -> pd.Series:
    '''Count most frequently purchased products.'''
    return df['product'].value_counts().head(n)


def build_rules_network(rules: pd.DataFrame, top_k: int = 30) -> nx.DiGraph:
    '''Build a directed graph from top association rules.'''
    G = nx.DiGraph()

    if rules.empty:
        return G

    subset = rules.head(top_k)

    for _, row in subset.iterrows():
        antecedents = list(row['antecedents'])
        consequents = list(row['consequents'])
        lift = row.get('lift', 1.0)
        confidence = row.get('confidence', 0.0)

        for a in antecedents:
            for c in consequents:
                G.add_edge(
                    a,
                    c,
                    lift=lift,
                    confidence=confidence,
                )
    return G


def graph_to_plotly_figure(G: nx.DiGraph) -> go.Figure:
    '''Convert a NetworkX DiGraph into an interactive Plotly network figure.'''
    if len(G.nodes) == 0:
        return go.Figure()

    pos = nx.spring_layout(G, k=0.6, iterations=50)

    # Edges
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=0.5),
        hoverinfo='none',
        mode='lines',
    )

    # Nodes
    node_x = []
    node_y = []
    node_text = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_text,
        textposition='top center',
        marker=dict(size=14),
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
    )
    return fig
