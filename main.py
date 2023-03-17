"""
Name: Raj KUmar Phagami
ID: C0846583
Module: 
Subject: Assignment 1
"""

# importing libraries
import os
import re, json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

#import NLP tools
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Conifguration keys
DEVELOPER_KEY = "AIzaSyDvVm7GvZC-gc6TxXWt9KIOvgBaCM7VI9I"
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"

# creating a youtube client
youtube_client = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=DEVELOPER_KEY)# Define function to calculate sentiment score using VADER

# Define function to get video data from YouTube API
def get_video_data(video_id):
    try:
        youtube_client = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=DEVELOPER_KEY)

        video_response = youtube_client.videos().list(
            part="snippet,contentDetails,statistics",
            id=video_id
        ).execute()
        data = [
            video_id,
            video_response['items'][0]['snippet']['description'],
            int(video_response['items'][0]['statistics']['viewCount']),
            int(video_response['items'][0]['statistics']['likeCount'] if 'likeCount' in video_response['items'][0]['statistics'] else 0),
            int(video_response['items'][0]['statistics']['dislikeCount'] if 'dislikeCount' in video_response['items'][0]['statistics'] else 0),
            int(video_response['items'][0]['statistics']['commentCount'] if 'dislikeCount' in video_response['items'][0]['statistics'] else 0),
            video_response['items'][0]['contentDetails']['duration'],
            int(video_response['items'][0]['statistics']['favoriteCount'])
        ]
        return data
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
# Define function to get video comments from YouTube API
def get_video_comments(video_id, max_results=100):
    try:
        comments = []
        next_page_token = None
        while len(comments) < max_results:
            comment_response = youtube_client.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=max_results,
                pageToken=next_page_token
            ).execute()
            for item in comment_response['items']:
                comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
                comments.append(comment)
            if 'nextPageToken' in comment_response:
                next_page_token = comment_response['nextPageToken']
            else:
                break
        return comments
    except Exception as e:
        print(f"An error occurred: {e}")
        return []
    
# filter the video ids from the dataframe that works with the YouTube API
#check if the video id is valid
#check connection
def check_connection(video_id):
    try:
        video_response = youtube_client.videos().list(
            part="snippet,statistics,contentDetails",
            id=video_id
        ).execute()
        print(f"Connection successful: {video_id}")
        return True
    except HttpError as e:
        print(f"Connection unsuccessful: {video_id}")
        return False
    
#get the video id ids that works with the YouTube API
def filter_video_ids(video_ids):
    valid_video_ids = []
    for video_id in video_ids:
        if check_connection(video_id):
            valid_video_ids.append(video_id)
    return valid_video_ids
    
   
#Calculate the sentiment score using VADER      
def get_sentiment_score(df):
    analyzer = SentimentIntensityAnalyzer()
    df['polarity_scores'] = df['comments_preprocessed'].apply(lambda x: analyzer.polarity_scores(x))
    df['compound'] = df['polarity_scores'].apply(lambda score_dict: score_dict['compound'])
    df['sentiment'] = df['compound'].apply(lambda c: 'pos' if c >=0 else 'neg')
    return df

#preprocessing the text data
def preprocessing(df):
    #lower string   
    df['comments_preprocessed'] = df['comments'].str.lower()
    #remove punctuation
    df['comments_preprocessed'] = df['comments_preprocessed'].str.replace('[^\w\s]','')
    #remove numbers
    df['comments_preprocessed'] = df['comments_preprocessed'].str.replace('\d+', '')
    #remove emojis
    df['comments_preprocessed'] = df['comments_preprocessed'].str.replace('[^\w\s#@/:%.,_-]', '', flags=re.UNICODE)
    #remove whitespace
    df['comments_preprocessed'] = df['comments_preprocessed'].str.strip()
    #tokenize the text using tokenizer
    df['comments_preprocessed'] = df['comments_preprocessed'].apply(lambda x: word_tokenize(x))
    #stemming
    df['comments_preprocessed'] = df['comments_preprocessed'].apply(lambda x: [PorterStemmer().stem(y) for y in x])
    #remove stopwords
    df['comments_preprocessed'] = df['comments_preprocessed'].apply(lambda x: ' '.join([word for word in x if word not in (stopwords.words('english'))]))
    return df

# Create a DataFrame from the comments dictionary id and comments
def get_comments(filtered_ids, new_columns):
    comments = {}
    for video in filtered_ids:
        comments[video] = get_video_comments(video)
    mappped_comments = []
    for key, value in comments.items():
        if value!=[]:
            for comment in value:
                mappped_comments.append([key, comment])
    df_comments = pd.DataFrame(mappped_comments, columns=new_columns)
    df_comments.to_csv('comments.csv', index=False)
    return df_comments

# Get data for each video and store in a list
def get_youtube_data(filtered_ids, columns):
    video_data_list = []
    for video in filtered_ids:
        video_data = get_video_data(video)
        if video_data is not None:
            video_data_list.append(video_data)
    df = pd.DataFrame(video_data_list,columns=columns)
    return df

# list of top 10 viewcount videos
def get_top_10_viewCount(df):
    top_10 = df.sort_values(by='viewCount', ascending=False).head(10)
    return top_10

#list of bottom 10 viewcount videos
def get_bottom_10_viewCount(df):
    bottom_10 = df.sort_values(by='viewCount', ascending=True).head(10)
    return bottom_10

#list of top 10 likecount videos
def get_top_10_likeCount(df):
    top_10 = df.sort_values(by='likeCount', ascending=False).head(10)
    return top_10

#list of bottom 10 likecount videos
def get_bottom_10_likeCount(df):
    bottom_10 = df.sort_values(by='likeCount', ascending=True).head(10)
    return bottom_10

#list of top 10 dislikecount videos
def get_top_10_dislikeCount(df):
    top_10 = df.sort_values(by='dislikeCount', ascending=False).head(10)
    return top_10

#Video with maximum duration
def get_max_duration(df):
    df['duration'] = df['duration'].apply(lambda x: pd.to_timedelta(x).total_seconds())
    max_duration = df.sort_values(by='duration', ascending=False).head(1)
    return max_duration

#get statistics
def get_stats(df):
    #List of top 10pvideos with most views
    top_10 = get_top_10_viewCount(df)
    print("******Top 10 videos with most views******")
    print(top_10.head(10))
    
    #List of top ten videos with least views
    bottom_10 = get_bottom_10_viewCount(df)
    print("******Bottom 10 videos with least views******")
    print(bottom_10.head(10))
    
    #List of top 10 videos with most likes
    top_10_likes = get_top_10_likeCount(df)
    print("******Top 10 videos with most likes******")
    print(top_10_likes.head(10))
    
    #List of bottom 10 with least likes
    bottom_10_likes = get_bottom_10_likeCount(df)
    print("******Bottom 10 videos with least likes******")
    print(bottom_10_likes.head(10))
    
    #Video with  maximum duration
    max_duration = get_max_duration(df)
    print("******Video with maximum duration******")
    print(max_duration)
    
def plot_bar(df):
    #plot bar graph for top 10 videos with most views
    top_10 = get_top_10_viewCount(df)
    top_10.plot.bar(x='youtubeId', y='viewCount', rot=0)
    plt.title('Top 10 videos with most views')
    plt.savefig('top_10_views.jpg')
    
    #plot bar graph for top 10 videos with most likes
    top_10_likes = get_bottom_10_viewCount(df)
    top_10_likes.plot.bar(x='youtubeId', y='viewCount', rot=0)
    plt.title('Top 10 videos with most views')
    plt.savefig('bottom_10_views.jpg')



if __name__ == '__main__':
    # Read CSV file into pandas DataFrame and extract youtube ids 
    print("Filtering video ids froim the csv file")
    ids = pd.read_csv('vdoLinks.csv')
    video_ids = ids['youtubeId'].tolist()
    
    #filter the video ids that works with the YouTube API
    #check if the file filtered_ids.json exists 
    if os.path.exists('filtered_ids.json'):
        with open('filtered_ids.json') as f:
            filtered_ids = json.load(f)
    else:
        filtered_ids = filter_video_ids(video_ids)
        with open('filtered_ids.json', 'w') as f:
            json.dump(filtered_ids, f)
    
    # Get data for each video and store in a dataframe
    columns = ['youtubeId', 'description', 'viewCount', 'likeCount', 'dislikeCount', 'commentCount', 'duration', 'favoriteCount']
    print("******Fetching Video data******")
    df = get_youtube_data(filtered_ids, columns)
    
    #Getting Statistics
    print("******Getting Statistics******")
    get_stats(df)
    
    #plot
    plot_bar(df)
    
    #getting comments for each video. if the file comments.csv exists, read from the file
    print("******Fetching comments******")
    if os.path.exists('comments.csv'):
        df_comments = pd.read_csv('comments.csv')
    else:
        new_columns = ['youtubeId', 'comments']
        df_comments = get_comments(filtered_ids, new_columns)
        df_comments.to_csv('comments.csv', index=False)
    
    #preprocessing comments
    preprocessed_df = preprocessing(df_comments)
    print("******Preprocessing comments******")
    print(preprocessed_df.head(10))
    
    #Sentinment Analysis
    print("******Vader Sentiment Analysis******")
    df_comments = get_sentiment_score(df_comments)
    print(df_comments.head(5))
    
    
    
    
    
