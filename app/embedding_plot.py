from sklearn.manifold import TSNE
import plotly.express as px
import pandas as pd
import torch
import plotly.graph_objects as go
from fastapi.responses import JSONResponse

def generate_plot(user_input, word2idx, embeddings):
    vocab_subset = list(word2idx.keys())[2:]
    selected_words = [word for word in user_input if word in vocab_subset]

    if not selected_words:
        return JSONResponse(content={
            "error": "No valid tokens found for visualization.",
            "data": [],
            "layout": {}
        })

    indices = [word2idx[word] for word in selected_words]
    vecs = embeddings[indices]

    n_samples = len(vecs)
    perplexity = min(30, max(3, n_samples - 1))  # ensures 3 ≤ perplexity < n_samples

    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    reduced = tsne.fit_transform(vecs)
    
    x_coords = reduced[:, 0].tolist()
    y_coords = reduced[:, 1].tolist()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=x_coords,
        y=y_coords,
        text=selected_words,
        mode='markers+text',
        textposition='top center',
        marker=dict(color='#636efa', symbol='circle'),
        hovertemplate="x=%{x}<br>y=%{y}<br>word=%{text}<extra></extra>",
        showlegend=False
    ))
    
    fig.update_layout(
        title="Token Embedding Visualization (t-SNE)",
        xaxis_title="x",
        yaxis_title="y"
    )
    
    # Convert to dict (not JSON) to preserve array structure
    payload_dict = fig.to_dict()
    
    return JSONResponse(content=payload_dict)