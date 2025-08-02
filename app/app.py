from fastapi import FastAPI, HTTPException
import torch
import torch.nn.functional as F
from src.model import FakeNewsClassifier  
from src.preprocess import clean_and_tokenize, encode_and_pad
import json
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.responses import HTMLResponse
from app.embedding_plot import generate_plot
import functools
import time

# Global variables for lazy loading
_model = None
_word2idx = None
_config = None
_device = None

def get_device():
    global _device
    if _device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _device

def load_model():
    """Lazy load the model only when needed"""
    global _model, _word2idx, _config
    
    if _model is None:
        start_time = time.time()
        device = get_device()
        
        try:
            checkpoint = torch.load("models/fake_news_checkpoint.pt", map_location=device)
            _config = checkpoint["config"]
            _word2idx = checkpoint["word2idx"]
            
            _model = FakeNewsClassifier(
                vocab_size=_config["vocab_size"],
                embed_dim=_config["embed_dim"],
                hidden_dim=_config["hidden_dim"],
                num_classes=_config["num_classes"],
                padding_idx=_config["padding_idx"],
                num_layers=_config["num_layers"],
                drop_out=_config["drop_out"],
                pretrained_embeddings=_config["pretrained_embeddings"]
            ).to(device)
            
            _model.load_state_dict(checkpoint["model_state"])
            _model.eval()
            
            print(f"Model loaded in {time.time() - start_time:.2f} seconds")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise HTTPException(status_code=500, detail="Model loading failed")
    
    return _model, _word2idx, _config

app = FastAPI(title="Fake News Detector", description="AI-powered fake news detection API")

app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.get("/")
def read_home():
    return FileResponse("app/static/index.html")

class NewsInput(BaseModel):
    title: str

@app.post("/predict")
def predict(input_data: NewsInput):
    try:
        model, word2idx, config = load_model()
        
        x = [clean_and_tokenize(input_data.title)]
        x = encode_and_pad(x, word2idx, max_length=200)
        x = torch.tensor(x, dtype=torch.long).to(get_device())
        
        with torch.no_grad():
            output = model(x)
            prediction = torch.argmax(output, dim=1).item()
        
        return prediction

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/embedding-plot", response_class=HTMLResponse)
def embedding_plot_snippet(input_data: NewsInput):
    try:
        model, word2idx, config = load_model()
        
        embeddings_np = config["pretrained_embeddings"]
        if isinstance(config["pretrained_embeddings"], torch.Tensor):
            embeddings_np = config["pretrained_embeddings"].detach().cpu().numpy()
        
        html_snippet = generate_plot(clean_and_tokenize(input_data.title), word2idx, embeddings_np)
        return html_snippet
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": _model is not None}