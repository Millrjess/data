#!/usr/bin/env python
# coding: utf-8

# # **TikTok Project**
# **Course 2 - Get Started with Python**

# Welcome to the TikTok Project!
# 
# You have just started as a data professional at TikTok.
# 
# The team is still in the early stages of the project. You have received notice that TikTok's leadership team has approved the project proposal. To gain clear insights to prepare for a claims classification model, TikTok's provided data must be examined to begin the process of exploratory data analysis (EDA).
# 
# A notebook was structured and prepared to help you in this project. Please complete the following questions.

# # **Course 2 End-of-course project: Inspect and analyze data**
# 
# In this activity, you will examine data provided and prepare it for analysis.
# <br/>
# 
# **The purpose** of this project is to investigate and understand the data provided. This activity will:
# 
# 1.   Acquaint you with the data
# 
# 2.   Compile summary information about the data
# 
# 3.   Begin the process of EDA and reveal insights contained in the data
# 
# 4.   Prepare you for more in-depth EDA, hypothesis testing, and statistical analysis
# 
# **The goal** is to construct a dataframe in Python, perform a cursory inspection of the provided dataset, and inform TikTok data team members of your findings.
# <br/>
# *This activity has three parts:*
# 
# **Part 1:** Understand the situation
# * How can you best prepare to understand and organize the provided TikTok information?
# 
# **Part 2:** Understand the data
# 
# * Create a pandas dataframe for data learning and future exploratory data analysis (EDA) and statistical activities
# 
# * Compile summary information about the data to inform next steps
# 
# **Part 3:** Understand the variables
# 
# * Use insights from your examination of the summary data to guide deeper investigation into variables
# 
# <br/>
# 
# To complete the activity, follow the instructions and answer the questions below. Then, you will us your responses to these questions and the questions included in the Course 2 PACE Strategy Document to create an executive summary.
# 
# Be sure to complete this activity before moving on to Course 3. You can assess your work by comparing the results to a completed exemplar after completing the end-of-course project.

# # **Identify data types and compile summary information**
# 

# Throughout these project notebooks, you'll see references to the problem-solving framework PACE. The following notebook components are labeled with the respective PACE stage: Plan, Analyze, Construct, and Execute.
# 
# # **PACE stages**
# 
# <img src="images/Pace.png" width="100" height="100" align=left>
# 
#    *        [Plan](#scrollTo=psz51YkZVwtN&line=3&uniqifier=1)
#    *        [Analyze](#scrollTo=mA7Mz_SnI8km&line=4&uniqifier=1)
#    *        [Construct](#scrollTo=Lca9c8XON8lc&line=2&uniqifier=1)
#    *        [Execute](#scrollTo=401PgchTPr4E&line=2&uniqifier=1)

# <img src="images/Plan.png" width="100" height="100" align=left>
# 
# 
# ## **PACE: Plan**
# 
# Consider the questions in your PACE Strategy Document and those below to craft your response:
# 
# 

# ### **Task 1. Understand the situation**
# 
# *   How can you best prepare to understand and organize the provided information?
# 
# 
# *Begin by exploring your dataset and consider reviewing the Data Dictionary.*

# ==> ENTER YOUR RESPONSE HERE

# <img src="images/Analyze.png" width="100" height="100" align=left>
# 
# ## **PACE: Analyze**
# 
# Consider the questions in your PACE Strategy Document to reflect on the Analyze stage.

# ### **Task 2a. Imports and data loading**
# 
# Start by importing the packages that you will need to load and explore the dataset. Make sure to use the following import statements:
# *   `import pandas as pd`
# 
# *   `import numpy as np`
# 

# In[1]:


# Import packages
import pandas as pd

import numpy as np


# Then, load the dataset into a dataframe. Creating a dataframe will help you conduct data manipulation, exploratory data analysis (EDA), and statistical activities.
# 
# **Note:** As shown in this cell, the dataset has been automatically loaded in for you. You do not need to download the .csv file, or provide more code, in order to access the dataset and proceed with this lab. Please continue with this activity by completing the following instructions.

# In[2]:


# Load dataset into dataframe
data = pd.read_csv("tiktok_dataset.csv")


# ### **Task 2b. Understand the data - Inspect the data**
# 
# View and inspect summary information about the dataframe by **coding the following:**
# 
# 1. `data.head(10)`
# 2. `data.info()`
# 3. `data.describe()`
# 
# *Consider the following questions:*
# 
# **Question 1:** When reviewing the first few rows of the dataframe, what do you observe about the data? What does each row represent?
# 
# **Question 2:** When reviewing the `data.info()` output, what do you notice about the different variables? Are there any null values? Are all of the variables numeric? Does anything else stand out?
# 
# **Question 3:** When reviewing the `data.describe()` output, what do you notice about the distributions of each variable? Are there any questionable values? Does it seem that there are outlier values?
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 

# In[3]:


# Display and examine the first ten rows of the dataframe
data.head(10)


# In[4]:


# Get summary info
data.info()


# In[5]:


# Get summary statistics
data.describe()


# Question 1: Each row in the data represents one TikTok video that includes details like the video’s content, duration, and how people interacted with it.
# 
# Question 2: Most variables are complete, but a few have missing values, and not all columns are numbers—some describe things like whether the video was verified or if the author was banned.
# 
# Question 3: Some videos got way more views, likes, or comments than others, showing that a few viral posts are driving up the averages.

# ### **Task 2c. Understand the data - Investigate the variables**
# 
# In this phase, you will begin to investigate the variables more closely to better understand them.
# 
# You know from the project proposal that the ultimate objective is to use machine learning to classify videos as either claims or opinions. A good first step towards understanding the data might therefore be examining the `claim_status` variable. Begin by determining how many videos there are for each different claim status.

# In[6]:


# What are the different values for claim status and how many of each are in the data?
data['claim_status'].value_counts(dropna=False)


# The number of claim and opinion videos is nearly equal, which is good for training a fair model, and only a small number of videos are missing a label, which means the dataset is mostly complete.

# Next, examine the engagement trends associated with each different claim status.
# 
# Start by using Boolean masking to filter the data according to claim status, then calculate the mean and median view counts for each claim status.

# In[7]:


# What is the average view count of videos with "claim" status?
import pandas as pd

# Load your data (replace 'your_dataset.csv' with the actual file path)
data = pd.read_csv("tiktok_dataset.csv")

# Filter data for claim videos
claims = data[data['claim_status'] == 'claim']

# Calculate the mean view count for claim videos
claim_mean_views = claims['video_view_count'].mean()
claim_mean_views


# In[8]:


# What is the average view count of videos with "opinion" status?
import pandas as pd

# Load your data (replace 'your_dataset.csv' with the actual file path)
data = pd.read_csv('tiktok_dataset.csv')

# Filter data for opinion videos
opinions = data[data['claim_status'] == 'opinion']

# Calculate the mean view count for opinion videos
opinion_mean_views = opinions['video_view_count'].mean()
opinion_mean_views


# **Question:** What do you notice about the mean and media within each claim category?
# 
# Now, examine trends associated with the ban status of the author.
# 
# Use `groupby()` to calculate how many videos there are for each combination of categories of claim status and author ban status.

# In[9]:


# Get counts for each group combination of claim status and author ban status
# Group by both 'claim_status' and 'author_ban_status' and count the number of videos in each group
videos_by_claim_and_ban = data.groupby(['claim_status', 'author_ban_status']).size()

# Display the result
videos_by_claim_and_ban


# **Question:** What do you notice about the number of claims videos with banned authors? Why might this relationship occur?
# 
# Continue investigating engagement levels, now focusing on `author_ban_status`.
# 
# Calculate the median video share count of each author ban status.

# In[10]:


# Group by 'author_ban_status' and calculate the median share count
median_share_by_ban_status = data.groupby('author_ban_status')['video_share_count'].median()

# Display the result
median_share_by_ban_status


# In[11]:


# What's the median video share count of each author ban status?
# Group by 'author_ban_status' and calculate the median share count
median_share_by_ban_status = data.groupby('author_ban_status')['video_share_count'].median()

# Display the result
median_share_by_ban_status


# **Question:** What do you notice about the share count of banned authors, compared to that of active authors? Explore this in more depth.
# 
# Use `groupby()` to group the data by `author_ban_status`, then use `agg()` to get the count, mean, and median of each of the following columns:
# * `video_view_count`
# * `video_like_count`
# * `video_share_count`
# 
# Remember, the argument for the `agg()` function is a dictionary whose keys are columns. The values for each column are a list of the calculations you want to perform.

# In[12]:


# Group by 'author_ban_status' and use agg() to calculate count, mean, and median for each specified column
agg_stats_by_ban_status = data.groupby('author_ban_status').agg({
    'video_view_count': ['count', 'mean', 'median'],
    'video_like_count': ['count', 'mean', 'median'],
    'video_share_count': ['count', 'mean', 'median']
})

# Display the result
agg_stats_by_ban_status


# **Question:** What do you notice about the number of views, likes, and shares for banned authors compared to active authors?
# 
# Now, create three new columns to help better understand engagement rates:
# * `likes_per_view`: represents the number of likes divided by the number of views for each video
# * `comments_per_view`: represents the number of comments divided by the number of views for each video
# * `shares_per_view`: represents the number of shares divided by the number of views for each video

# In[13]:


# Create a likes_per_view column
data['likes_per_view'] = data['video_like_count'] / data['video_view_count']

# Create a comments_per_view column
data['comments_per_view'] = data['video_comment_count'] / data['video_view_count']

# Create a shares_per_view column
data['shares_per_view'] = data['video_share_count'] / data['video_view_count']

# Display the first few rows to confirm the new columns
data[['video_like_count', 'video_view_count', 'video_comment_count', 'video_share_count', 'likes_per_view', 'comments_per_view', 'shares_per_view']].head()


# Use `groupby()` to compile the information in each of the three newly created columns for each combination of categories of claim status and author ban status, then use `agg()` to calculate the count, the mean, and the median of each group.

# In[14]:


# Group by claim_status and author_ban_status, then aggregate the necessary statistics
agg_stats = data.groupby(['claim_status', 'author_ban_status']).agg(
    count_likes=('likes_per_view', 'count'),
    mean_likes=('likes_per_view', 'mean'),
    median_likes=('likes_per_view', 'median'),
    count_comments=('comments_per_view', 'count'),
    mean_comments=('comments_per_view', 'mean'),
    median_comments=('comments_per_view', 'median'),
    count_shares=('shares_per_view', 'count'),
    mean_shares=('shares_per_view', 'mean'),
    median_shares=('shares_per_view', 'median')
).reset_index()

# Display the aggregated statistics
print(agg_stats)


# **Question:**
# 
# How does the data for claim videos and opinion videos compare or differ? Consider views, comments, likes, and shares.
# 
# Claim videos generally have higher engagement (views, likes, comments, and shares) than opinion videos, likely due to their more impactful or controversial content.
# 

# <img src="images/Construct.png" width="100" height="100" align=left>
# 
# ## **PACE: Construct**
# 
# **Note**: The Construct stage does not apply to this workflow. The PACE framework can be adapted to fit the specific requirements of any project.
# 
# 
# 

# <img src="images/Execute.png" width="100" height="100" align=left>
# 
# ## **PACE: Execute**
# 
# Consider the questions in your PACE Strategy Document and those below to craft your response.

# ### **Given your efforts, what can you summarize for Rosie Mae Bradshaw and the TikTok data team?**
# 
# *Note for Learners: Your answer should address TikTok's request for a summary that covers the following points:*
# 
# *   What percentage of the data is comprised of claims and what percentage is comprised of opinions?
# *   What factors correlate with a video's claim status?
# *   What factors correlate with a video's engagement level?
# 

# Summary for Rosie Mae Bradshaw and the TikTok Data Team:
# 
# Claim vs. Opinion: About 49.2% of the videos are claims, 48.5% are opinions, and 2.3% have missing claim status data.
# 
# Factors Correlating with Claim Status: Videos from banned or under review authors are more likely to be claims, and claim videos usually have higher views and shares.
# 
# Factors Correlating with Engagement: Claim videos get more engagement than opinion videos. Banned authors also see more likes and shares per view, indicating higher interaction.
# 
# In summary, claim videos, especially from banned authors, attract more engagement and views compared to opinion videos.
# 
# ______________
# 
# Simplified for CEO and laypersons 
# 
# Glossary:
# 
# Claim Videos: Videos that present information or arguments, often trying to persuade the viewer or convey factual content.
# 
# Opinion Videos: Videos that express personal thoughts, views, or feelings, typically about a topic, without aiming to present factual evidence.
# 
# Claim Status: The classification of a video as either a claim or an opinion, indicating whether the video provides factual content or personal opinion.
# 
# Engagement: The level of interaction with a video, measured by likes, shares, comments, and views.
# 
# Author Ban Status: Indicates whether the video author is active, banned, or under review by the platform.
# 
# Likes Per View: A metric that measures the number of likes a video receives relative to the number of views.
# 
# Shares Per View: A metric that measures how often a video is shared relative to its number of views.
# 
# Comments Per View: A metric that measures how many comments a video receives relative to its number of views.
# 
# Banned Authors: Creators whose accounts have been suspended due to violating platform rules.
# 
# Under Review Authors: Creators whose content or account is being evaluated for possible policy violations.
# 
# Active Authors: Creators whose accounts are in good standing, without restrictions or suspensions.
# 

# **Congratulations!** You've completed this lab. However, you may not notice a green check mark next to this item on Coursera's platform. Please continue your progress regardless of the check mark. Just click on the "save" icon at the top of this notebook to ensure your work has been logged.
