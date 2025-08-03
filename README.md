# Fake News Detector

An AI-powered fake news detection system built with PyTorch and FastAPI. This project uses a bidirectional LSTM neural network with pre-trained word embeddings to classify news articles as either real or fake.

## Overview

This system analyzes news headlines and articles to determine their authenticity using deep learning techniques. It combines natural language processing with neural networks to provide real-time fake news detection capabilities.

## Model Architecture

The fake news detector employs a sophisticated neural network architecture:

- **Embedding Layer**: Pre-trained GloVe word embeddings (100-dimensional vectors)
- **Bidirectional LSTM**: 2-layer LSTM with hidden dimension of 32, capturing contextual information from both directions
- **Dropout**: 50% dropout for regularization and preventing overfitting
- **Classification Head**: Linear layer with softmax activation for binary classification
- **Output**: Binary classification (0: Real news, 1: Fake news)

## Word Embeddings & NLP

The system leverages pre-trained GloVe embeddings to understand semantic relationships between words:

- **GloVe Embeddings**: 100-dimensional vectors trained on large text corpora
- **Vocabulary Building**: Custom vocabulary built from the training dataset
- **Text Preprocessing**: Tokenization, cleaning, and sequence padding
- **Semantic Understanding**: Captures word meanings and relationships

## Nearest Neighbors Analysis

The system implements cosine similarity-based nearest neighbors to explore word relationships:

- **Cosine Similarity**: Computes similarity between word vectors in the embedding space
- **Top-K Neighbors**: Finds the 5 most similar words for each input token
- **Semantic Clustering**: Groups semantically related words together
- **Interactive Hover Information**: Displays nearest neighbors when hovering over words in visualizations
- **Vocabulary Exploration**: Helps understand how the model interprets word meanings and relationships

## Visualizations

The project includes interactive embedding visualizations:

- **Word Embedding Plots**: Interactive 3D visualizations of word vectors using UMAP dimensionality reduction
- **UMAP Dimensionality Reduction**: Converts high-dimensional embeddings to 3D for visualization while preserving local structure
- **Interactive Plotly Charts**: Dynamic plots showing word relationships and clusters
- **Real-time Visualization**: Generate embedding plots for any input text
- **Nearest Neighbors Display**: Hover over words to see their most similar neighbors in the vocabulary

## Technical Stack

- **Deep Learning**: PyTorch with LSTM networks
- **Web Framework**: FastAPI for high-performance API
- **Visualization**: Plotly for interactive charts
- **NLP**: NLTK for text processing
- **Data Processing**: Pandas and NumPy
- **Dimensionality Reduction**: UMAP for embedding visualization
- **Deployment**: Heroku-ready with Procfile

## Project Structure

```
fake-news-detector/
├── app/                    # Web application
│   ├── app.py             # FastAPI application
│   ├── embedding_plot.py  # Embedding visualization
│   └── static/            # Static files
├── src/                   # Source code
│   ├── model.py           # Neural network model
│   ├── preprocess.py      # Data preprocessing
│   └── train.py           # Training script
├── models/                # Trained models
├── data/                  # Datasets
└── requirements.txt       # Dependencies
```

## Key Features

- **Real-time Classification**: Instant fake news detection
- **Embedding Visualization**: Interactive word vector plots with nearest neighbors
- **Pre-trained Model**: Ready-to-use trained neural network
- **Scalable Architecture**: Production-ready deployment
- **Comprehensive Dataset**: Large-scale real and fake news training data

## Quick Start

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the application**
   ```bash
   uvicorn app.app:app --reload
   ```

3. **Access the web interface**
   Navigate to `http://localhost:8000`

## Dataset

The model is trained on comprehensive datasets:
- **Real News**: Articles from legitimate news sources
- **Fake News**: Articles identified as fake or misleading
- **Pre-trained Embeddings**: GloVe embeddings for enhanced word representation

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for improvements.

---

**Note**: This is a demonstration project. For production use, additional validation and security measures should be implemented. 