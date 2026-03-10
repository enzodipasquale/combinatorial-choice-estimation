import sys
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go

HOVER = 'θ₀=%{x:.2f}<br>θ₁=%{y:.2f}<br>obj=%{z:.4f}<extra></extra>'


def load(path):
    d = np.load(path)
    return d["t1"], d["t2"], d["obj"], d["theta_true"], float(d["beta"])


def add_surface(fig, t1, t2, obj, theta_true, beta, row, col):
    ij = np.unravel_index(obj.argmin(), obj.shape)
    fig.add_trace(go.Surface(x=t1, y=t2, z=obj, colorscale='Viridis', opacity=0.9,
                             showscale=False, hovertemplate=HOVER), row=row, col=col)
    fig.add_trace(go.Scatter3d(
        x=[theta_true[0]], y=[theta_true[1]], z=[0],
        mode='markers', marker=dict(size=5, color='red'),
        name=f'true ({theta_true[0]}, {theta_true[1]})', showlegend=(col == 1),
    ), row=row, col=col)
    fig.add_trace(go.Scatter3d(
        x=[t1[ij[1]]], y=[t2[ij[0]]], z=[obj.min()],
        mode='markers', marker=dict(size=5, color='white', line=dict(width=1, color='black')),
        name=f'min ({t1[ij[1]]:.2f}, {t2[ij[0]]:.2f})', showlegend=(col == 1),
    ), row=row, col=col)


files = sys.argv[1:] if len(sys.argv) > 1 else ["grid_results.npz"]

if len(files) == 1:
    t1, t2, obj, theta_true, beta = load(files[0])
    fig = go.Figure()
    ij = np.unravel_index(obj.argmin(), obj.shape)
    fig.add_trace(go.Surface(x=t1, y=t2, z=obj, colorscale='Viridis', opacity=0.9,
                             hovertemplate=HOVER))
    fig.add_trace(go.Scatter3d(
        x=[theta_true[0]], y=[theta_true[1]], z=[0],
        mode='markers', marker=dict(size=6, color='red'),
        name=f'true ({theta_true[0]}, {theta_true[1]})'))
    fig.add_trace(go.Scatter3d(
        x=[t1[ij[1]]], y=[t2[ij[0]]], z=[obj.min()],
        mode='markers', marker=dict(size=6, color='white', line=dict(width=1, color='black')),
        name=f'grid min ({t1[ij[1]]:.2f}, {t2[ij[0]]:.2f})'))
    fig.update_layout(
        title=f'Objective landscape (β={beta:.4f})',
        scene=dict(xaxis_title='θ₀', yaxis_title='θ₁', zaxis_title='obj'),
        width=900, height=700)
    fig.write_html("obj_landscape_3d.html")
    print("Saved obj_landscape_3d.html")

elif len(files) == 2:
    data = [load(f) for f in files]
    fig = make_subplots(rows=1, cols=2,
                        specs=[[{"type": "scene"}, {"type": "scene"}]],
                        subplot_titles=[f'β={d[4]:.4f}' for d in data])
    for i, (t1, t2, obj, theta_true, beta) in enumerate(data):
        add_surface(fig, t1, t2, obj, theta_true, beta, row=1, col=i + 1)

    axis = dict(xaxis_title='θ₀', yaxis_title='θ₁', zaxis_title='obj')
    fig.update_layout(width=1600, height=700, title='Objective landscape comparison',
                      scene=axis, scene2=axis)
    fig.write_html("obj_landscape_comparison.html")
    print("Saved obj_landscape_comparison.html")
