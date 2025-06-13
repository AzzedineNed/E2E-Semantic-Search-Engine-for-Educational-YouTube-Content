import numpy as np
import polars
import sentence_transformers
import sklearn
from sklearn.metrics import DistanceMetric
import os

# Available channels
AVAILABLE_CHANNELS = [
    'freecodecamp',
    'netninja', 
    'krishnaik',
    'jeffheaton',
    'firstprinciples',
    '3blue1brown',
    'techworldnana',
    'yousuckatprogramming',
    'theconstructsim'
]

def returnSearchResultIndexes(query: str, 
                              df: polars.DataFrame,
                              model, 
                              dist: DistanceMetric) -> np.ndarray:
    """
    Function to return indexes of top search results for a specific channel
    """
    
    # embed query
    query_embedding = model.encode(query).reshape(1, -1)
    
    # compute distances between query and titles/descriptions
    dist_arr = dist.pairwise(df[:, 4:388].to_numpy(), query_embedding) + dist.pairwise(df[:, 388:772].to_numpy(), query_embedding)

    # search parameters
    threshold = 40  # eye-balled threshold for manhattan distance
    top_k = 20

    # evaluate videos close to query based on threshold
    idx_below_threshold = np.argwhere(dist_arr.flatten() < threshold).flatten()
    
    if len(idx_below_threshold) == 0:
        # If no results below threshold, return top k closest
        idx_sorted = np.argsort(dist_arr.flatten())[:top_k]
        return idx_sorted
    
    # keep top k closest videos
    idx_sorted = np.argsort(dist_arr[idx_below_threshold], axis=0).flatten()

    # return indexes of search results
    return idx_below_threshold[idx_sorted][:top_k]

def getAvailableChannels():
    """Return list of available channels with existing indexes."""
    available = []
    for channel in AVAILABLE_CHANNELS:
        index_path = f'app/data/{channel}-index.parquet'
        if os.path.exists(index_path):
            available.append(channel)
    return available