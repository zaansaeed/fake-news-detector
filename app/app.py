from flask import Flask, request, jsonify
import torch
import torch.nn.functional as F
from model import FakeNewsModel  # Your model class
from preprocess import preprocess_text  # Your preprocessing/tokenization
import json