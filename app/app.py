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
from app.embedding_plot import generate_plot  # from the file above


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


checkpoint = torch.load("models/fake_news_checkpoint.pt", map_location=device)
config = checkpoint["config"]
model = FakeNewsClassifier(
    vocab_size=config["vocab_size"],
    embed_dim=config["embed_dim"],
    hidden_dim=config["hidden_dim"],
    num_classes=config["num_classes"],
    padding_idx=config["padding_idx"],
    num_layers=config["num_layers"],
    drop_out=config["drop_out"],
    pretrained_embeddings=config["pretrained_embeddings"]
).to(device)
word2idx = checkpoint["word2idx"]
model.load_state_dict(checkpoint["model_state"])
model.eval()


app = FastAPI()



app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.get("/")
def read_home():
    
    return FileResponse("app/static/index.html")

class NewsInput(BaseModel):
    title: str

@app.post("/predict")
def predict(input_data: NewsInput):
    try:
        x = [clean_and_tokenize(input_data.title)]
       
        x = encode_and_pad(x, word2idx, max_length=200)
        x = torch.tensor(x,dtype=torch.long).to(device)  # Make sure it returns a tensor
        with torch.no_grad():
            output = model(x)
            prediction = torch.argmax(output, dim=1).item()
            labels = {0: "fake", 1: "real"}
        return {"prediction": labels[prediction]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    



@app.post("/embedding-plot", response_class=HTMLResponse)
def embedding_plot_snippet(input_data: NewsInput):
    embeddings_np = config["pretrained_embeddings"]
    if isinstance(config["pretrained_embeddings"], torch.Tensor):
        embeddings_np = config["pretrained_embeddings"].detach().cpu().numpy()
    html_snippet = generate_plot(clean_and_tokenize(input_data.title),word2idx, embeddings_np)
    return html_snippet


