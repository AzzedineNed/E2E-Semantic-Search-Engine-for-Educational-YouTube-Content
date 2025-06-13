import requests
import json
import polars as pl
from sentence_transformers import SentenceTransformer
import os

# Channel configuration with their YouTube channel IDs
CHANNELS = {
    'freecodecamp': 'UC8butISFwT-Wl7EV0hUK0BQ',
    'netninja': 'UCW5YeuERMmlnqo4oq8vwUpg',
    'krishnaik': 'UCNU_lfiiWBdtULKOw6X0Dig',
    'jeffheaton': 'UCR1-GEpyOPzT2AO4D_eifdw',
    'firstprinciples': 'UCf0WB91t8Ky6AuYcQV0CcLw',
    '3blue1brown': 'UCYO_jab_esuFRV4b17AJtAw',
    'techworldnana': 'UCdngmbVKX1Tgre699-XLlUA',
    'yousuckatprogramming': 'UCMN0a7GHQnC6H74SmCGSmdw',
    'theconstructsim': 'UCt6Lag-vv25fTX3e11mVY1Q'
}

def getVideoRecords(response: requests.models.Response) -> list:
    """Extracts video metadata from YouTube API response."""
    video_record_list = []
    data = json.loads(response.text)

    for item in data.get('items', []):
        video_record = {
            'video_id': item['contentDetails']['videoId'],
            'datetime': item['contentDetails']['videoPublishedAt'],
            'title': item['snippet']['title'],
            'description': item['snippet'].get('description', 'n/a')
        }
        video_record_list.append(video_record)

    return video_record_list

def getChannelVideos(channel_name: str):
    """Fetches all videos from a specific YouTube channel with descriptions."""
    api_key = os.getenv('YT_API_KEY')
    channel_id = CHANNELS[channel_name]
    base_url = 'https://www.googleapis.com/youtube/v3/playlistItems'

    # Get Uploads Playlist ID
    playlist_response = requests.get(
        f"https://www.googleapis.com/youtube/v3/channels?part=contentDetails&id={channel_id}&key={api_key}"
    )
    if playlist_response.status_code != 200:
        print(f"Channel API Error for {channel_name}: {playlist_response.text}")
        return

    playlist_id = json.loads(playlist_response.text)['items'][0]['contentDetails']['relatedPlaylists']['uploads']
    print(f"Processing {channel_name} - Playlist ID: {playlist_id}")

    page_token = None
    video_record_list = []

    while True:
        params = {
            "key": api_key,
            "playlistId": playlist_id,
            "part": "contentDetails,snippet",
            "maxResults": 50,
            "pageToken": page_token
        }

        response = requests.get(base_url, params=params)
        if response.status_code != 200:
            print(f"Playlist Items API Error for {channel_name}: {response.text}")
            break

        new_videos = getVideoRecords(response)
        video_record_list.extend(new_videos)
        print(f"Fetched {len(video_record_list)} videos from {channel_name}...")

        data = json.loads(response.text)
        page_token = data.get('nextPageToken', None)
        if not page_token:
            break

    df = pl.DataFrame(video_record_list)
    df.write_parquet(f'app/data/{channel_name}-videos.parquet')
    print(f"✅ Saved {len(video_record_list)} videos for {channel_name}")

def setDatatypes(df: pl.DataFrame) -> pl.DataFrame:
    """Sets proper data types for DataFrame columns."""
    df = df.with_columns(pl.col('datetime').cast(pl.Datetime))
    return df

def transformChannelData(channel_name: str):
    """Preprocesses and cleans the video data for a specific channel."""
    df = pl.read_parquet(f'app/data/{channel_name}-videos.parquet')
    df = setDatatypes(df)
    df.write_parquet(f'app/data/{channel_name}-videos.parquet')

def createChannelEmbeddings(channel_name: str):
    """Generates embeddings for video titles and descriptions for a specific channel."""
    df = pl.read_parquet(f'app/data/{channel_name}-videos.parquet')
    
    # Filter out videos without descriptions
    df = df.filter(pl.col("description") != "n/a")
    
    if len(df) == 0:
        print(f"No valid videos with descriptions for {channel_name}, skipping...")
        return
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    column_name_list = ['title', 'description']

    for column_name in column_name_list:
        embedding_arr = model.encode(df[column_name].to_list())
        
        # Create embedding columns
        schema_dict = {f"{column_name}_embedding_{i}": pl.Float64 for i in range(embedding_arr.shape[1])}
        df_embeddings = pl.DataFrame(embedding_arr, schema=schema_dict)
        
        df = df.hstack(df_embeddings)

    df.write_parquet(f'app/data/{channel_name}-index.parquet')
    print(f"✅ Created embeddings for {channel_name}: {len(df)} videos")

def processAllChannels():
    """Process all configured channels."""
    os.makedirs('app/data', exist_ok=True)
    
    for channel_name in CHANNELS.keys():
        print(f"\nProcessing channel: {channel_name}")
        print("=" * 50)
        
        try:
            getChannelVideos(channel_name)
            transformChannelData(channel_name)
            createChannelEmbeddings(channel_name)
        except Exception as e:
            print(f"❌ Error processing {channel_name}: {str(e)}")
            continue
    
    print("\n✅ All channels processed!")
