from sklearn.manifold import TSNE
import plotly.express as px
import pandas as pd
import torch
import plotly.graph_objects as go
from fastapi.responses import JSONResponse
import numpy as np
from functools import lru_cache
import time

# Cache for t-SNE results to avoid recomputing for same words
_tsne_cache = {}

def get_embedding_subset(user_input, word2idx, embeddings, max_words=20):
    """Get a subset of embeddings for faster t-SNE computation"""
    vocab_subset = list(word2idx.keys())[2:]  # Skip PAD and UNK tokens
    selected_words = [word for word in user_input if word in vocab_subset]
    
    # Limit the number of words for faster processing
    if len(selected_words) > max_words:
        selected_words = selected_words[:max_words]
    
    return selected_words

@lru_cache(maxsize=100)
def compute_tsne_for_words(word_tuple, embeddings_np):
    """Cache t-SNE results for repeated word combinations"""
    words = list(word_tuple)
    indices = [word2idx[word] for word in words]
    vecs = embeddings_np[indices]
    
    n_samples = len(vecs)
    if n_samples < 2:
        return None, None, words
    
    # Optimize perplexity based on sample size
    perplexity = min(30, max(3, min(n_samples - 1, n_samples // 2)))
    
    # Use faster t-SNE parameters for real-time visualization
    tsne = TSNE(
        n_components=2, 
        perplexity=perplexity, 
        random_state=42,
        n_iter=1000,  # Reduced iterations for speed
        method='barnes_hut' if n_samples > 50 else 'exact'  # Faster for larger datasets
    )
    
    start_time = time.time()
    reduced = tsne.fit_transform(vecs)
    print(f"t-SNE computed in {time.time() - start_time:.2f} seconds for {n_samples} words")
    
    return reduced[:, 0].tolist(), reduced[:, 1].tolist(), words

def generate_plot(user_input, word2idx, embeddings):
    """Generate optimized t-SNE visualization"""
    try:
        # Convert embeddings to numpy if needed
        if isinstance(embeddings, torch.Tensor):
            embeddings_np = embeddings.detach().cpu().numpy()
        else:
            embeddings_np = embeddings
        
        # Get word subset for faster processing
        selected_words = get_embedding_subset(user_input, word2idx, embeddings_np, max_words=15)
        
        if not selected_words:
            return JSONResponse(content={
                "error": "No valid tokens found for visualization.",
                "data": [],
                "layout": {}
            })
        
        # Create cache key
        word_tuple = tuple(sorted(selected_words))
        
        # Check cache first
        if word_tuple in _tsne_cache:
            x_coords, y_coords, words = _tsne_cache[word_tuple]
        else:
            # Compute t-SNE
            x_coords, y_coords, words = compute_tsne_for_words(word_tuple, embeddings_np)
            if x_coords is not None:
                _tsne_cache[word_tuple] = (x_coords, y_coords, words)
        
        if x_coords is None:
            return JSONResponse(content={
                "error": "Insufficient data for t-SNE visualization.",
                "data": [],
                "layout": {}
            })
        
        # Create optimized plot
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=x_coords,
            y=y_coords,
            text=words,
            mode='markers+text',
            textposition='top center',
            marker=dict(
                color='#636efa', 
                symbol='circle',
                size=10,
                line=dict(width=1, color='#2c3e50')
            ),
            textfont=dict(size=10),
            hovertemplate="<b>%{text}</b><br>x: %{x:.3f}<br>y: %{y:.3f}<extra></extra>",
            showlegend=False
        ))
        
        fig.update_layout(
            title={
                'text': f"Token Embedding Visualization (t-SNE) - {len(words)} words",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 16}
            },
            xaxis_title="t-SNE Dimension 1",
            yaxis_title="t-SNE Dimension 2",
            margin=dict(l=50, r=50, t=80, b=50),
            height=500,
            width=700,
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray',
                zeroline=False
            ),
            yaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray',
                zeroline=False
            )
        )
        
        # Convert to dict for JSON response
        payload_dict = fig.to_dict()
        
        return JSONResponse(content=payload_dict)
        
    except Exception as e:
        print(f"Error in generate_plot: {e}")
        return JSONResponse(content={
            "error": f"Visualization failed: {str(e)}",
            "data": [],
            "layout": {}
        })

# Cache management function
def clear_tsne_cache():
    """Clear the t-SNE cache to free memory"""
    global _tsne_cache
    _tsne_cache.clear()
    compute_tsne_for_words.cache_clear()
    print("t-SNE cache cleared")