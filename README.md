# YoutubeVideoPopularityPredictor
Motivation is to help a huge set of people to predict the popularity of their next video. Influencers can modify the content accordingly and generate greater revenue from the views.

## Dataset
Dataset was downloaded from the YOUTUBE - 8M
dataset. The dataset of cookery videos was saved locally.

Considering features with numerical value only, the following attributes were decided:
<I> Duration, LikeCount PreviousVideo, DislikeCount PreviousVideo, CommentCount PreviousVideo, NumberOfTags, ChannelSubscribers, ChannelUploads, ChannelViews, VideoLength, DescriptionLength, SocialLinks<I>
  
##  Thumbnail of Videos
The thumbnail is an important feature for the predictions and hence we extracted and used thumbnails of all videos using the video ids. 

## Features after thumbnail extraction:
<I>Duration, LikeCount PreviousVideo, DislikeCount PreviousVideo, CommentCount PreviousVideo, NumberOfTags, ChannelSubscribers, ChannelUploads, ChannelViews, VideoLength, DescriptionLength, SocialLinks, Thumbnail Result<I>
  
## Training the Models - I
Initially, the data have been trained on 4 models:<br>
• Linear Regression<br>
• Decision Tree Regression<br>
• Random Forest Regression<br>
• Support Vector Machine<br>

The results (accuracies) from these models were not up to the mark and thus thumbnail is introduced in II part to improve the output.

## Training the Models - II
1) The first part focuses on extracting the feature, ”thumbnail result”. Convolutional Neural Network has been used for the same. Dataset for the CNN consists of the Thumbnails and the labelled views.
2) The ”thumbnail result” feature is now used with the above-mentioned features and has been trained with the following models, with the target variable being the actual number of views.<br>
• Support Vector Regressor<br>
• Random Forest Regressor<br>
