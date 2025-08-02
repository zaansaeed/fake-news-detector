"""
Performance configuration for the Fake News Detector
"""
import os

# Model loading settings
LAZY_LOAD_MODEL = True
MODEL_LOAD_TIMEOUT = 30  # seconds

# T-SNE visualization settings
MAX_WORDS_FOR_TSNE = 15
TSNE_CACHE_SIZE = 100
TSNE_N_ITER = 1000  # Reduced for speed
TSNE_PERPLEXITY_MAX = 30

# API settings
MAX_REQUEST_SIZE = 1024 * 1024  # 1MB
REQUEST_TIMEOUT = 30  # seconds

# Frontend settings
DEBOUNCE_DELAY = 1000  # milliseconds
PLOT_DELAY = 500  # milliseconds

# Memory management
CLEAR_CACHE_THRESHOLD = 50  # Clear cache when it reaches this size
MAX_CONCURRENT_REQUESTS = 10

# Development settings
DEBUG_MODE = os.getenv("DEBUG", "False").lower() == "true"
LOG_PERFORMANCE = DEBUG_MODE

# Performance monitoring
ENABLE_PERFORMANCE_LOGGING = True
PERFORMANCE_LOG_INTERVAL = 100  # Log every 100 requests 