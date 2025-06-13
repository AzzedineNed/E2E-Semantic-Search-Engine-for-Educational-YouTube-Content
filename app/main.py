from fastapi import FastAPI, HTTPException
import polars as pl
from sentence_transformers import SentenceTransformer
from sklearn.metrics import DistanceMetric
import numpy as np
from app.search_function import returnSearchResultIndexes, getAvailableChannels
import os

# define model info
model_name = 'all-MiniLM-L6-v2'

# load model once at startup
model = SentenceTransformer(model_name)

# create distance metric object
dist_name = 'manhattan'
dist = DistanceMetric.get_metric(dist_name)

# create FastAPI object
app = FastAPI()

# Global dictionary to cache loaded channel indexes
channel_indexes = {}

def loadChannelIndex(channel_name: str):
    """Load channel index with caching."""
    if channel_name not in channel_indexes:
        index_path = f'app/data/{channel_name}-index.parquet'
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index file not found for channel: {channel_name}")
        channel_indexes[channel_name] = pl.read_parquet(index_path)
    return channel_indexes[channel_name]

# API operations
@app.get("/")
def health_check():
    return {'health_check': 'OK', 'version': '2.0.0'}

@app.get("/info")
def info():
    return {
        'name': 'multi-channel-yt-search', 
        'description': "Search API for multiple educational YouTube channels",
        'available_channels': getAvailableChannels()
    }

@app.get("/channels")
def list_channels():
    return {'available_channels': getAvailableChannels()}

@app.get("/search/{channel}")
def search(channel: str, query: str):
    # Validate channel
    available_channels = getAvailableChannels()
    if channel not in available_channels:
        raise HTTPException(
            status_code=400, 
            detail=f"Channel '{channel}' not available. Available: {available_channels}"
        )
    
    # Load channel index
    try:
        df = loadChannelIndex(channel)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Index not found for channel: {channel}")
    
    # Get search results
    idx_result = returnSearchResultIndexes(query, df, model, dist)
    idx_result_list = idx_result.tolist()
    
    # Select relevant columns
    result_df = df.select(['title', 'video_id', 'datetime'])[idx_result_list]
    result_dict = result_df.to_dict(as_series=False)
    
    return {
        'query': query,
        'channel': channel,
        'results': result_dict,
        'total_results': len(result_dict['title'])
    }