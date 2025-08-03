import numpy as np
import plotly.graph_objects as go
from fastapi.responses import JSONResponse

# Robust import for umap-learn
try:
    from umap import UMAP
except Exception:
    from umap.umap_ import UMAP

def generate_plot(user_input, word2idx, embeddings, k=5, max_tokens=60):
    # Select known tokens (ordered, deduped, capped)
    selected_words, seen = [], set()
    for w in user_input:
        if w in word2idx and w not in seen:
            selected_words.append(w); seen.add(w)
            if len(selected_words) >= max_tokens:
                break

    if len(selected_words) < 4:
        return JSONResponse(content={
            "error": "Need at least 4 known tokens for visualization.",
            "data": [], "layout": {}
        })

    # Indices for selected tokens
    indices = np.array([word2idx[w] for w in selected_words], dtype=np.int64)

    # Ensure embeddings are float32 NumPy
    if hasattr(embeddings, "detach"):  # torch.Tensor
        E = embeddings.detach().cpu().numpy().astype(np.float32, copy=False)
    else:
        E = np.asarray(embeddings, dtype=np.float32)

    vecs = E[indices]  # (n, d)

    # ---- Nearest neighbors in ORIGINAL embedding space (cosine) ----
    norms = np.linalg.norm(E, axis=1, keepdims=True)
    E_norm = E / np.maximum(norms, 1e-12)
    S = E_norm[indices]                # (n, d)
    sim = S @ E_norm.T                 # (n, |V|) cosine similarity

    idx2word = {v: k for k, v in word2idx.items()}
    hover_texts = []
    for i, src_idx in enumerate(indices):
        # Top-k over full vocab; exclude self
        cand = np.argpartition(sim[i], -(k+1))[-(k+1):]
        cand = cand[np.argsort(sim[i, cand])][::-1]
        cand = [c for c in cand if c != src_idx][:k]
        nbr_words = [idx2word.get(c, f"<{c}>") for c in cand]
        hover_texts.append(f"<b>{selected_words[i]}</b><br>NN: " + ", ".join(nbr_words))

    # ---- 3D UMAP reduction (fast, good local structure) ----
    n = len(selected_words)
    # Ensure 2 ≤ n_neighbors ≤ n-1 to avoid spectral eigendecomp errors
    n_neighbors = min(15, max(5, n // 2))
    n_neighbors = max(2, min(n_neighbors, n - 1))

    # For very small n, avoid spectral init path that can call eigh with bad k
    init_mode = "random" if n < 10 else "spectral"

    reducer = UMAP(
        n_components=3,
        n_neighbors=n_neighbors,
        min_dist=0.1,
        metric="cosine",     # cosine works well for embeddings
        random_state=42,
        init=init_mode,
    )
    reduced = reducer.fit_transform(vecs)  # (n, 3)

    x, y, z = reduced[:, 0].tolist(), reduced[:, 1].tolist(), reduced[:, 2].tolist()

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        text=selected_words,        # label next to point
        hovertext=hover_texts,      # rich hover with nearest neighbors
        hoverinfo="text",
        mode="markers+text",
        textposition="top center",
        marker=dict(symbol="circle"),
        showlegend=False
    ))

    fig.update_layout(
        title="Token Embedding Visualization (UMAP, 3D) + Nearest Neighbors",
        scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z"),
        margin=dict(l=10, r=10, t=40, b=10)
    )

    return JSONResponse(content=fig.to_dict())
