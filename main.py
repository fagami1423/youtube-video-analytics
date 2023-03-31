"""
Name: Raj KUmar Phagami
ID: C0846583
Module: 2023@_AML 3204_2 Social Media Analytics
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

from config import DEVELOPER_KEY, YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION

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
        if len(video_response['items']) == 0:
            print(f"Video not found: {video_id}")
            return []
        else:
            data = [
                video_id,
                video_response['items'][0]['snippet']['title'],
                video_response['items'][0]['snippet']['description'],
                int(video_response['items'][0]['statistics']['viewCount'] if 'viewCount' in video_response['items'][0]['statistics'] else 0),
                int(video_response['items'][0]['statistics']['likeCount'] if 'likeCount' in video_response['items'][0]['statistics'] else 0),
                int(video_response['items'][0]['statistics']['dislikeCount'] if 'dislikeCount' in video_response['items'][0]['statistics'] else 0),
                int(video_response['items'][0]['statistics']['commentCount'] if 'dislikeCount' in video_response['items'][0]['statistics'] else 0),
                video_response['items'][0]['contentDetails']['duration'],
                int(video_response['items'][0]['statistics']['favoriteCount'])
            ]
            return data

    except HttpError as e:
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
    except HttpError as e:
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
    df['comments_preprocessed'] = df['comments_preprocessed'].str.replace('[^\w\s]','', regex=True)
    #remove numbers
    df['comments_preprocessed'] = df['comments_preprocessed'].str.replace('\d+', '', regex=True)
    #remove emojis
    df['comments_preprocessed'] = df['comments_preprocessed'].str.replace('[^\w\s#@/:%.,_-]', '', regex=True)
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
    try:
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
    except HttpError as e:
        print(f"An error occurred: {e}")
        return None


# Get data for each video and store in a list
def get_youtube_data(filtered_ids, columns):
    video_data_list = []
    for index,video in enumerate(filtered_ids):
        video_data = get_video_data(video)
        if video_data is not None:
            video_data_list.append(video_data)
        if index == 100:
            break
    df = pd.DataFrame(video_data_list,columns=columns)
    df.to_csv('video_data.csv', index=False)
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
    # df['duration'] = df['duration'].apply(lambda x: pd.to_timedelta(x).total_seconds())
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
    plt.figure(figsize=(10, 6))  # set figure size
    plt.barh(top_10['title'], top_10['viewCount'])
    plt.xlabel('Video Title', fontsize=12)  # set x-axis label
    plt.ylabel('View Count', fontsize=12)  # set y-axis label
    plt.title('Top 10 Most Viewed Videos', fontsize=14)
    plt.xticks(rotation=90)  # rotate x-axis labels
    plt.subplots_adjust(left=0.4)  # adjust bottom margin
    plt.savefig('top_10_views.jpg')
    
    #plot bar graph for top 10 videos with least views
    top_10_views = get_bottom_10_viewCount(df)
    plt.figure(figsize=(10, 6))  # set figure size
    plt.barh(top_10_views['title'], top_10_views['viewCount'])
    plt.xlabel('Video Title', fontsize=12)  # set x-axis label
    plt.ylabel('View Count', fontsize=12)  # set y-axis label
    plt.title('Bottom 10 Viewed Videos', fontsize=14)
    plt.xticks(rotation=90)  # rotate x-axis labels
    plt.subplots_adjust(left=0.4)  # adjust bottom margin
    plt.savefig('bottom_10_views.jpg')
    
def plot_sentiments(df):
    #plot bar graph for top 10 videos with most positive compound sentiment using matplotlib
    top_10_compound = df.sort_values(by='compound', ascending=False).head(10)
    top_10_compound['short_comment'] = top_10_compound['comments_preprocessed'].apply(lambda x: ' '.join(x.split()[:10]))
    
    plt.figure(figsize=(15, 6))
    plt.barh(top_10_compound['short_comment'],top_10_compound['compound'])
    plt.xlabel('Compound Sentiment')
    plt.ylabel('Comment')
    plt.subplots_adjust(left=0.5) 
    plt.title('Top 10 videos with most positive compound sentiment')
    plt.savefig('top_10_compound.jpg')
    

if __name__ == '__main__':
    # Read CSV file into pandas DataFrame and extract youtube ids 
    print("Filtering video ids froim the csv file")
    ids = pd.read_csv('vdoLinks.csv')
    video_ids = ids['youtubeId'].tolist()
    
    # #ck if the file filtered_ids.json exists 
    # if os.path.exists('filtered_ids.json'):
    #     with open('filtered_ids.json') as f:
    #         filtered_ids = json.load(f)
    # else:
    #     filtered_ids = filter_video_ids(video_ids)
    #     with open('filtered_ids.json', 'w') as f:
    #         json.dump(filtered_ids, f)
    
    # Get data for each video and store in a dataframe
    columns = ['youtubeId','title','description', 'viewCount', 'likeCount', 'dislikeCount', 'commentCount', 'duration', 'favoriteCount']
    print("******Fetching Video data******")
    df = get_youtube_data(video_ids, columns)
    
    #Getting Statistics
    print("******Getting Statistics******")
    get_stats(df)
    
    #plotting bar graphs for top 10 videos with most views and most likes and save the figure to the disk
    plot_bar(df)
    
    #getting comments for each video. if the file comments.csv exists, read from the file
    print("******Fetching comments******")
    if os.path.exists('comments.csv'):
        df_comments = pd.read_csv('comments.csv')
    else:
        new_columns = ['youtubeId','comments']
        df_comments = get_comments(df['youtubeId'].to_list(), new_columns)
        df_comments.to_csv('comments.csv', index=False)
    
    #preprocessing comments
    preprocessed_df = preprocessing(df_comments)
    print("******Preprocessing comments******")
    print(preprocessed_df.head(10))
    
    #Sentinment Analysis
    print("******Vader Sentiment Analysis******")
    
    plot_sentiments(df_comments)
    
    sentiment_counts = df_comments.groupby(['youtubeId', 'sentiment']).size().reset_index(name='counts')
    sorted_counts = sentiment_counts.sort_values(['youtubeId', 'counts'], ascending=[True, False])
    
    positive_counts = sorted_counts[sorted_counts['sentiment'] == 'pos']

    # select top 10 counts
    # top_10_counts = positive_counts.groupby('youtubeId').head(10)
    # group by youtubeId and sum counts
    positive_counts_by_youtubeId = positive_counts.groupby('youtubeId').sum().reset_index()
    df_comments = get_sentiment_score(df_comments)
    # sort by counts and select top 10 youtubeIds
    top_10_youtubeIds = positive_counts_by_youtubeId.sort_values('counts', ascending=False).head(10)

    # plot youtubeId vs counts
    plt.figure(figsize=(10,6))
    plt.bar(top_10_youtubeIds['youtubeId'], top_10_youtubeIds['counts'])
    plt.xticks(rotation=90)
    plt.xlabel('YouTube ID')
    plt.ylabel('Sentiment Counts')
    plt.title('Top 10 YouTube IDs with Positive Sentiment Counts')
    plt.savefig("Top_10_vidoes_with_positive.jpg")
    print(sorted_counts)
    print(df_comments.head(5))
    
    
    
    
    
