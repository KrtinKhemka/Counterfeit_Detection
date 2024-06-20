from transformers import pipeline
import numpy as np 
import pandas as pd


sent_pipeline = pipeline("sentiment-analysis")