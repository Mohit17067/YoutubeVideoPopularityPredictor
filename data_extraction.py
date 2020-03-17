#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import urllib, json
import subprocess


# In[2]:


# - iterate over tensorflow record, one example at a time
num_bad_labels = 0
num_videos_in_record = 0

import glob
filenames = glob.glob('*.tfrecord')
featureDict = {}

for video_level_data in filenames:
    print('processing ',video_level_data)
    
    num_videos_in_part = 0
    for example in tf.compat.v1.python_io.tf_record_iterator(video_level_data):
        num_videos_in_record += 1
        num_videos_in_part += 1
        # yield example
        tf_example = tf.train.Example.FromString(example)
        
        vid_id = tf_example.features.feature['id'].bytes_list.value[0].decode(encoding='UTF-8')
        # get list of labels from example
        label_idx_list = tf_example.features.feature["labels"].int64_list.value

        # instantiate list for KEY=vid_id
        featureDict[vid_id] = label_idx_list
        

    print('############# {} VIDEOS IN RECORD ##############'.format(num_videos_in_part))                    
print('\nAnalyzed {} videos from YT8M'.format(num_videos_in_record))


# In[3]:


# categories =  {22 : "Cooking",
#               11 : "Food",
#               29 : "Cuisine",
#               32 : "Recipe",
#               52 : "Dish",
#               58 : "Cuisine",
#               1567 : "Cook",
#               589 : "Grilling",
#               620 : "Barbecue",
#               808 : "Breakfast"}

# # label 204 is gym
count = 0
gymVids = set()
for val in featureDict:
    print(val)
    try:

    # get video id from example
    #print(type(tf_example.features.feature['video_id'].bytes_list.value))

        url = "data.yt8m.org/2/j/i/" + str(val[:2]) + "/" + str(val) + ".js"


        res = subprocess.check_output(["wget", url, "-q", "-O", "-"])
        res = str(res)
        vid_id = res[12:23]
        print(vid_id)
        gymVids.add(vid_id)
        count += 1
    except:
        print("e")
print()
print()
print("Total Count :" + str(count))
print(len(list(gymVids)))





import csv
import json
import urllib
import strict_rfc3339
import datetime
import calendar
import re
import math

api_key_1 = 'AIzaSyA5I4cDoxpBgfOX0JkWPXs_xURpFQ35nnY'
api_key_2 = 'AIzaSyDenAqx8_YUKvpAxXX0JmXIDHYd1S_9TZU'
api_key_3 = 'AIzaSyCzmkalWWnVlDJ2PBBNSfVomQN7vE6MDYw'
api_key_4 = 'AIzaSyDCsViDqvRWQU9G7dn_Y2_qleaqxd7sh8o'
api = [api_key_1,api_key_2,api_key_3,api_key_4]

firstTime = True
with open('videoStats.csv', 'a',newline='') as c:

    writer = csv.writer(c)
    
    if firstTime:
        writer.writerow(['Title', 
                         'Description',
                         'CategoryId',
                         'PublishedAt',
                         'CurrentTime',
                         'Life',
                         'Definition',
                         'Caption',
                         'Duration',
                         'Dimension',
                         'LikeCount', 
                         'DislikeCount', 
                         'ViewCount', 
                         'FavoriteCount', 
                         'CommentCount', 
                         'PublishedAt',
                         'Tags',
                         'ThumbnailDefault',
                         'ChannelId', 
                         'ChannelTitle',
                         'ChannelSubscribers',
                         'ChannelUploads'])
        firstTime = False

    gymVids = list(gymVids)
    counter = 0;
    print(len(gymVids))
    for ind,vid in enumerate(gymVids):
        api_key = api[ind//3000]
        try:
            url = "https://www.googleapis.com/youtube/v3/videos?id=" + vid + "&key=" + api_key + "&part=status,statistics,contentDetails,snippet"
            response = urllib.request.urlopen(url).read()
            data = json.loads(response)
            all_data = data['items']

            ChannelId = all_data[0]['snippet']['channelId']
            ChannelTitle = all_data[0]['snippet']['channelTitle']
            Title = all_data[0]['snippet']['title']
            Description = all_data[0]['snippet']['description']
            CategoryId = all_data[0]['snippet']['categoryId']
            Tags = all_data[0]['snippet']['tags']
            Thumbnail = all_data[0]['snippet']['thumbnails']['default']['url']
            publishedAt = all_data[0]['snippet']['publishedAt'] 
            PublishedAt	= int(strict_rfc3339.rfc3339_to_timestamp(publishedAt))
            currentTime	= datetime.datetime.utcnow() # current time as rtf3339
            currentTime	= datetime.datetime.timetuple(currentTime) # current time as timetuple
            CurrentTime	= calendar.timegm(currentTime) # current time as epoch timestamp
            Life = CurrentTime - PublishedAt

            Definition = all_data[0]['contentDetails']['definition']
            Caption = all_data[0]['contentDetails']['caption']
            #licensedContent = all_data[0]['contentDetails']['licensedContent']
            Dimension = all_data[0]['contentDetails']['dimension']

            duration = all_data[0]['contentDetails']['duration']
            duration_w = re.search(r"(\d+)w", duration, re.I)
            duration_w = int(duration_w.group(1)) if duration_w else 0
            duration_d = re.search(r"(\d+)d", duration, re.I)
            duration_d = int(duration_d.group(1)) if duration_d else 0
            duration_h = re.search(r"(\d+)h", duration, re.I)
            duration_h = int(duration_h.group(1)) if duration_h else 0
            duration_m = re.search(r"(\d+)m", duration, re.I)
            duration_m = int(duration_m.group(1)) if duration_m else 0
            duration_s = re.search(r"(\d+)s", duration, re.I)
            duration_s = int(duration_s.group(1)) if duration_s else 0
            duration = 0
            duration += duration_w * 7 * 24 * 60 * 60
            duration += duration_d * 24 * 60 * 60
            duration += duration_h * 60 * 60
            duration += duration_m * 60
            duration += duration_s * 1
            Duration = duration
            
#             duration = int(all_data[0]['fileDetails']['durationMs'])
#             Duration = duration // 1000
#             WidthPixels = int(all_data[0]['fileDetails']['videoStreams[]']['widthPixels'])
#             HeightPixels = int(all_data[0]['fileDetails']['videoStreams[]']['heightPixels'])

            CommentCount = int(all_data[0]['statistics']['commentCount'])
            ViewCount = int(all_data[0]['statistics']['viewCount'])
            FavoriteCount = int(all_data[0]['statistics']['favoriteCount'])
            LikeCount = int(all_data[0]['statistics']['likeCount'])
            DislikeCount = int(all_data[0]['statistics']['dislikeCount'])
            
            channel_url = "https://www.googleapis.com/youtube/v3/channels?id=" + str(ChannelId) + "&key=" + str(api_key) + "&part=statistics"
            channel_response = urllib.request.urlopen(channel_url).read()
            channel_data = json.loads(channel_response)
            all_cdata = channel_data['items']

            ChannelSubscribers = int(all_cdata[0]['statistics']['subscriberCount'])
            ChannelUploads = int(all_cdata[0]['statistics']['videoCount'])


            writer.writerow([Title, 
                             Description,
                             CategoryId,
                             PublishedAt,
                             CurrentTime,
                             Life,
                             Definition,
                             Caption,
                             Duration,
                             Dimension,
                             LikeCount, 
                             DislikeCount, 
                             ViewCount, 
                             FavoriteCount, 
                             CommentCount, 
                             PublishedAt,
                             Tags,
                             Thumbnail,
                             ChannelId, 
                             ChannelTitle,
                             ChannelSubscribers,
                             ChannelUploads])
            counter+=1
            print('done_video')
        except:
            print('skiped_video')
    print("Downloaded Videos :" + str(counter))






