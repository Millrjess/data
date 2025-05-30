# **TikTok Project**
**Course 2 - Get Started with Python**

Welcome to the TikTok Project!

You have just started as a data professional at TikTok.

The team is still in the early stages of the project. You have received notice that TikTok's leadership team has approved the project proposal. To gain clear insights to prepare for a claims classification model, TikTok's provided data must be examined to begin the process of exploratory data analysis (EDA).

A notebook was structured and prepared to help you in this project. Please complete the following questions.

# **Course 2 End-of-course project: Inspect and analyze data**

In this activity, you will examine data provided and prepare it for analysis.
<br/>

**The purpose** of this project is to investigate and understand the data provided. This activity will:

1.   Acquaint you with the data

2.   Compile summary information about the data

3.   Begin the process of EDA and reveal insights contained in the data

4.   Prepare you for more in-depth EDA, hypothesis testing, and statistical analysis

**The goal** is to construct a dataframe in Python, perform a cursory inspection of the provided dataset, and inform TikTok data team members of your findings.
<br/>
*This activity has three parts:*

**Part 1:** Understand the situation
* How can you best prepare to understand and organize the provided TikTok information?

**Part 2:** Understand the data

* Create a pandas dataframe for data learning and future exploratory data analysis (EDA) and statistical activities

* Compile summary information about the data to inform next steps

**Part 3:** Understand the variables

* Use insights from your examination of the summary data to guide deeper investigation into variables

<br/>

To complete the activity, follow the instructions and answer the questions below. Then, you will us your responses to these questions and the questions included in the Course 2 PACE Strategy Document to create an executive summary.

Be sure to complete this activity before moving on to Course 3. You can assess your work by comparing the results to a completed exemplar after completing the end-of-course project.

# **Identify data types and compile summary information**


Throughout these project notebooks, you'll see references to the problem-solving framework PACE. The following notebook components are labeled with the respective PACE stage: Plan, Analyze, Construct, and Execute.

# **PACE stages**

<img src="images/Pace.png" width="100" height="100" align=left>

   *        [Plan](#scrollTo=psz51YkZVwtN&line=3&uniqifier=1)
   *        [Analyze](#scrollTo=mA7Mz_SnI8km&line=4&uniqifier=1)
   *        [Construct](#scrollTo=Lca9c8XON8lc&line=2&uniqifier=1)
   *        [Execute](#scrollTo=401PgchTPr4E&line=2&uniqifier=1)

<img src="images/Plan.png" width="100" height="100" align=left>


## **PACE: Plan**

Consider the questions in your PACE Strategy Document and those below to craft your response:



### **Task 1. Understand the situation**

*   How can you best prepare to understand and organize the provided information?


*Begin by exploring your dataset and consider reviewing the Data Dictionary.*

==> ENTER YOUR RESPONSE HERE

<img src="images/Analyze.png" width="100" height="100" align=left>

## **PACE: Analyze**

Consider the questions in your PACE Strategy Document to reflect on the Analyze stage.

### **Task 2a. Imports and data loading**

Start by importing the packages that you will need to load and explore the dataset. Make sure to use the following import statements:
*   `import pandas as pd`

*   `import numpy as np`



```python
# Import packages
import pandas as pd

import numpy as np
```

Then, load the dataset into a dataframe. Creating a dataframe will help you conduct data manipulation, exploratory data analysis (EDA), and statistical activities.

**Note:** As shown in this cell, the dataset has been automatically loaded in for you. You do not need to download the .csv file, or provide more code, in order to access the dataset and proceed with this lab. Please continue with this activity by completing the following instructions.


```python
# Load dataset into dataframe
data = pd.read_csv("tiktok_dataset.csv")
```

### **Task 2b. Understand the data - Inspect the data**

View and inspect summary information about the dataframe by **coding the following:**

1. `data.head(10)`
2. `data.info()`
3. `data.describe()`

*Consider the following questions:*

**Question 1:** When reviewing the first few rows of the dataframe, what do you observe about the data? What does each row represent?

**Question 2:** When reviewing the `data.info()` output, what do you notice about the different variables? Are there any null values? Are all of the variables numeric? Does anything else stand out?

**Question 3:** When reviewing the `data.describe()` output, what do you notice about the distributions of each variable? Are there any questionable values? Does it seem that there are outlier values?


















```python
# Display and examine the first ten rows of the dataframe
data.head(10)

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>#</th>
      <th>claim_status</th>
      <th>video_id</th>
      <th>video_duration_sec</th>
      <th>video_transcription_text</th>
      <th>verified_status</th>
      <th>author_ban_status</th>
      <th>video_view_count</th>
      <th>video_like_count</th>
      <th>video_share_count</th>
      <th>video_download_count</th>
      <th>video_comment_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>claim</td>
      <td>7017666017</td>
      <td>59</td>
      <td>someone shared with me that drone deliveries a...</td>
      <td>not verified</td>
      <td>under review</td>
      <td>343296.0</td>
      <td>19425.0</td>
      <td>241.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>claim</td>
      <td>4014381136</td>
      <td>32</td>
      <td>someone shared with me that there are more mic...</td>
      <td>not verified</td>
      <td>active</td>
      <td>140877.0</td>
      <td>77355.0</td>
      <td>19034.0</td>
      <td>1161.0</td>
      <td>684.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>claim</td>
      <td>9859838091</td>
      <td>31</td>
      <td>someone shared with me that american industria...</td>
      <td>not verified</td>
      <td>active</td>
      <td>902185.0</td>
      <td>97690.0</td>
      <td>2858.0</td>
      <td>833.0</td>
      <td>329.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>claim</td>
      <td>1866847991</td>
      <td>25</td>
      <td>someone shared with me that the metro of st. p...</td>
      <td>not verified</td>
      <td>active</td>
      <td>437506.0</td>
      <td>239954.0</td>
      <td>34812.0</td>
      <td>1234.0</td>
      <td>584.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>claim</td>
      <td>7105231098</td>
      <td>19</td>
      <td>someone shared with me that the number of busi...</td>
      <td>not verified</td>
      <td>active</td>
      <td>56167.0</td>
      <td>34987.0</td>
      <td>4110.0</td>
      <td>547.0</td>
      <td>152.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>claim</td>
      <td>8972200955</td>
      <td>35</td>
      <td>someone shared with me that gross domestic pro...</td>
      <td>not verified</td>
      <td>under review</td>
      <td>336647.0</td>
      <td>175546.0</td>
      <td>62303.0</td>
      <td>4293.0</td>
      <td>1857.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>claim</td>
      <td>4958886992</td>
      <td>16</td>
      <td>someone shared with me that elvis presley has ...</td>
      <td>not verified</td>
      <td>active</td>
      <td>750345.0</td>
      <td>486192.0</td>
      <td>193911.0</td>
      <td>8616.0</td>
      <td>5446.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>claim</td>
      <td>2270982263</td>
      <td>41</td>
      <td>someone shared with me that the best selling s...</td>
      <td>not verified</td>
      <td>active</td>
      <td>547532.0</td>
      <td>1072.0</td>
      <td>50.0</td>
      <td>22.0</td>
      <td>11.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>claim</td>
      <td>5235769692</td>
      <td>50</td>
      <td>someone shared with me that about half of the ...</td>
      <td>not verified</td>
      <td>active</td>
      <td>24819.0</td>
      <td>10160.0</td>
      <td>1050.0</td>
      <td>53.0</td>
      <td>27.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>claim</td>
      <td>4660861094</td>
      <td>45</td>
      <td>someone shared with me that it would take a 50...</td>
      <td>verified</td>
      <td>active</td>
      <td>931587.0</td>
      <td>171051.0</td>
      <td>67739.0</td>
      <td>4104.0</td>
      <td>2540.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Get summary info
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 19382 entries, 0 to 19381
    Data columns (total 12 columns):
     #   Column                    Non-Null Count  Dtype  
    ---  ------                    --------------  -----  
     0   #                         19382 non-null  int64  
     1   claim_status              19084 non-null  object 
     2   video_id                  19382 non-null  int64  
     3   video_duration_sec        19382 non-null  int64  
     4   video_transcription_text  19084 non-null  object 
     5   verified_status           19382 non-null  object 
     6   author_ban_status         19382 non-null  object 
     7   video_view_count          19084 non-null  float64
     8   video_like_count          19084 non-null  float64
     9   video_share_count         19084 non-null  float64
     10  video_download_count      19084 non-null  float64
     11  video_comment_count       19084 non-null  float64
    dtypes: float64(5), int64(3), object(4)
    memory usage: 1.8+ MB



```python
# Get summary statistics
data.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>#</th>
      <th>video_id</th>
      <th>video_duration_sec</th>
      <th>video_view_count</th>
      <th>video_like_count</th>
      <th>video_share_count</th>
      <th>video_download_count</th>
      <th>video_comment_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>19382.000000</td>
      <td>1.938200e+04</td>
      <td>19382.000000</td>
      <td>19084.000000</td>
      <td>19084.000000</td>
      <td>19084.000000</td>
      <td>19084.000000</td>
      <td>19084.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>9691.500000</td>
      <td>5.627454e+09</td>
      <td>32.421732</td>
      <td>254708.558688</td>
      <td>84304.636030</td>
      <td>16735.248323</td>
      <td>1049.429627</td>
      <td>349.312146</td>
    </tr>
    <tr>
      <th>std</th>
      <td>5595.245794</td>
      <td>2.536440e+09</td>
      <td>16.229967</td>
      <td>322893.280814</td>
      <td>133420.546814</td>
      <td>32036.174350</td>
      <td>2004.299894</td>
      <td>799.638865</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>1.234959e+09</td>
      <td>5.000000</td>
      <td>20.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>4846.250000</td>
      <td>3.430417e+09</td>
      <td>18.000000</td>
      <td>4942.500000</td>
      <td>810.750000</td>
      <td>115.000000</td>
      <td>7.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>9691.500000</td>
      <td>5.618664e+09</td>
      <td>32.000000</td>
      <td>9954.500000</td>
      <td>3403.500000</td>
      <td>717.000000</td>
      <td>46.000000</td>
      <td>9.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>14536.750000</td>
      <td>7.843960e+09</td>
      <td>47.000000</td>
      <td>504327.000000</td>
      <td>125020.000000</td>
      <td>18222.000000</td>
      <td>1156.250000</td>
      <td>292.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>19382.000000</td>
      <td>9.999873e+09</td>
      <td>60.000000</td>
      <td>999817.000000</td>
      <td>657830.000000</td>
      <td>256130.000000</td>
      <td>14994.000000</td>
      <td>9599.000000</td>
    </tr>
  </tbody>
</table>
</div>



Question 1: Each row in the data represents one TikTok video that includes details like the video’s content, duration, and how people interacted with it.

Question 2: Most variables are complete, but a few have missing values, and not all columns are numbers—some describe things like whether the video was verified or if the author was banned.

Question 3: Some videos got way more views, likes, or comments than others, showing that a few viral posts are driving up the averages.

### **Task 2c. Understand the data - Investigate the variables**

In this phase, you will begin to investigate the variables more closely to better understand them.

You know from the project proposal that the ultimate objective is to use machine learning to classify videos as either claims or opinions. A good first step towards understanding the data might therefore be examining the `claim_status` variable. Begin by determining how many videos there are for each different claim status.


```python
# What are the different values for claim status and how many of each are in the data?
data['claim_status'].value_counts(dropna=False)

```




    claim      9608
    opinion    9476
    NaN         298
    Name: claim_status, dtype: int64



The number of claim and opinion videos is nearly equal, which is good for training a fair model, and only a small number of videos are missing a label, which means the dataset is mostly complete.

Next, examine the engagement trends associated with each different claim status.

Start by using Boolean masking to filter the data according to claim status, then calculate the mean and median view counts for each claim status.


```python
# What is the average view count of videos with "claim" status?
import pandas as pd

# Load your data (replace 'your_dataset.csv' with the actual file path)
data = pd.read_csv("tiktok_dataset.csv")

# Filter data for claim videos
claims = data[data['claim_status'] == 'claim']

# Calculate the mean view count for claim videos
claim_mean_views = claims['video_view_count'].mean()
claim_mean_views


```




    501029.4527477102




```python
# What is the average view count of videos with "opinion" status?
import pandas as pd

# Load your data (replace 'your_dataset.csv' with the actual file path)
data = pd.read_csv('tiktok_dataset.csv')

# Filter data for opinion videos
opinions = data[data['claim_status'] == 'opinion']

# Calculate the mean view count for opinion videos
opinion_mean_views = opinions['video_view_count'].mean()
opinion_mean_views


```




    4956.43224989447



**Question:** What do you notice about the mean and media within each claim category?

Now, examine trends associated with the ban status of the author.

Use `groupby()` to calculate how many videos there are for each combination of categories of claim status and author ban status.


```python
# Get counts for each group combination of claim status and author ban status
# Group by both 'claim_status' and 'author_ban_status' and count the number of videos in each group
videos_by_claim_and_ban = data.groupby(['claim_status', 'author_ban_status']).size()

# Display the result
videos_by_claim_and_ban


```




    claim_status  author_ban_status
    claim         active               6566
                  banned               1439
                  under review         1603
    opinion       active               8817
                  banned                196
                  under review          463
    dtype: int64



**Question:** What do you notice about the number of claims videos with banned authors? Why might this relationship occur?

Continue investigating engagement levels, now focusing on `author_ban_status`.

Calculate the median video share count of each author ban status.


```python
# Group by 'author_ban_status' and calculate the median share count
median_share_by_ban_status = data.groupby('author_ban_status')['video_share_count'].median()

# Display the result
median_share_by_ban_status
```




    author_ban_status
    active            437.0
    banned          14468.0
    under review     9444.0
    Name: video_share_count, dtype: float64




```python
# What's the median video share count of each author ban status?
# Group by 'author_ban_status' and calculate the median share count
median_share_by_ban_status = data.groupby('author_ban_status')['video_share_count'].median()

# Display the result
median_share_by_ban_status

```




    author_ban_status
    active            437.0
    banned          14468.0
    under review     9444.0
    Name: video_share_count, dtype: float64



**Question:** What do you notice about the share count of banned authors, compared to that of active authors? Explore this in more depth.

Use `groupby()` to group the data by `author_ban_status`, then use `agg()` to get the count, mean, and median of each of the following columns:
* `video_view_count`
* `video_like_count`
* `video_share_count`

Remember, the argument for the `agg()` function is a dictionary whose keys are columns. The values for each column are a list of the calculations you want to perform.


```python
# Group by 'author_ban_status' and use agg() to calculate count, mean, and median for each specified column
agg_stats_by_ban_status = data.groupby('author_ban_status').agg({
    'video_view_count': ['count', 'mean', 'median'],
    'video_like_count': ['count', 'mean', 'median'],
    'video_share_count': ['count', 'mean', 'median']
})

# Display the result
agg_stats_by_ban_status

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="3" halign="left">video_view_count</th>
      <th colspan="3" halign="left">video_like_count</th>
      <th colspan="3" halign="left">video_share_count</th>
    </tr>
    <tr>
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>median</th>
      <th>count</th>
      <th>mean</th>
      <th>median</th>
      <th>count</th>
      <th>mean</th>
      <th>median</th>
    </tr>
    <tr>
      <th>author_ban_status</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>active</th>
      <td>15383</td>
      <td>215927.039524</td>
      <td>8616.0</td>
      <td>15383</td>
      <td>71036.533836</td>
      <td>2222.0</td>
      <td>15383</td>
      <td>14111.466164</td>
      <td>437.0</td>
    </tr>
    <tr>
      <th>banned</th>
      <td>1635</td>
      <td>445845.439144</td>
      <td>448201.0</td>
      <td>1635</td>
      <td>153017.236697</td>
      <td>105573.0</td>
      <td>1635</td>
      <td>29998.942508</td>
      <td>14468.0</td>
    </tr>
    <tr>
      <th>under review</th>
      <td>2066</td>
      <td>392204.836399</td>
      <td>365245.5</td>
      <td>2066</td>
      <td>128718.050339</td>
      <td>71204.5</td>
      <td>2066</td>
      <td>25774.696999</td>
      <td>9444.0</td>
    </tr>
  </tbody>
</table>
</div>



**Question:** What do you notice about the number of views, likes, and shares for banned authors compared to active authors?

Now, create three new columns to help better understand engagement rates:
* `likes_per_view`: represents the number of likes divided by the number of views for each video
* `comments_per_view`: represents the number of comments divided by the number of views for each video
* `shares_per_view`: represents the number of shares divided by the number of views for each video


```python
# Create a likes_per_view column
data['likes_per_view'] = data['video_like_count'] / data['video_view_count']

# Create a comments_per_view column
data['comments_per_view'] = data['video_comment_count'] / data['video_view_count']

# Create a shares_per_view column
data['shares_per_view'] = data['video_share_count'] / data['video_view_count']

# Display the first few rows to confirm the new columns
data[['video_like_count', 'video_view_count', 'video_comment_count', 'video_share_count', 'likes_per_view', 'comments_per_view', 'shares_per_view']].head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>video_like_count</th>
      <th>video_view_count</th>
      <th>video_comment_count</th>
      <th>video_share_count</th>
      <th>likes_per_view</th>
      <th>comments_per_view</th>
      <th>shares_per_view</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>19425.0</td>
      <td>343296.0</td>
      <td>0.0</td>
      <td>241.0</td>
      <td>0.056584</td>
      <td>0.000000</td>
      <td>0.000702</td>
    </tr>
    <tr>
      <th>1</th>
      <td>77355.0</td>
      <td>140877.0</td>
      <td>684.0</td>
      <td>19034.0</td>
      <td>0.549096</td>
      <td>0.004855</td>
      <td>0.135111</td>
    </tr>
    <tr>
      <th>2</th>
      <td>97690.0</td>
      <td>902185.0</td>
      <td>329.0</td>
      <td>2858.0</td>
      <td>0.108282</td>
      <td>0.000365</td>
      <td>0.003168</td>
    </tr>
    <tr>
      <th>3</th>
      <td>239954.0</td>
      <td>437506.0</td>
      <td>584.0</td>
      <td>34812.0</td>
      <td>0.548459</td>
      <td>0.001335</td>
      <td>0.079569</td>
    </tr>
    <tr>
      <th>4</th>
      <td>34987.0</td>
      <td>56167.0</td>
      <td>152.0</td>
      <td>4110.0</td>
      <td>0.622910</td>
      <td>0.002706</td>
      <td>0.073175</td>
    </tr>
  </tbody>
</table>
</div>



Use `groupby()` to compile the information in each of the three newly created columns for each combination of categories of claim status and author ban status, then use `agg()` to calculate the count, the mean, and the median of each group.


```python
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


```

      claim_status author_ban_status  count_likes  mean_likes  median_likes  \
    0        claim            active         6566    0.329542      0.326538   
    1        claim            banned         1439    0.345071      0.358909   
    2        claim      under review         1603    0.327997      0.320867   
    3      opinion            active         8817    0.219744      0.218330   
    4      opinion            banned          196    0.206868      0.198483   
    5      opinion      under review          463    0.226394      0.228051   
    
       count_comments  mean_comments  median_comments  count_shares  mean_shares  \
    0            6566       0.001393         0.000776          6566     0.065456   
    1            1439       0.001377         0.000746          1439     0.067893   
    2            1603       0.001367         0.000789          1603     0.065733   
    3            8817       0.000517         0.000252          8817     0.043729   
    4             196       0.000434         0.000193           196     0.040531   
    5             463       0.000536         0.000293           463     0.044472   
    
       median_shares  
    0       0.049279  
    1       0.051606  
    2       0.049967  
    3       0.032405  
    4       0.030728  
    5       0.035027  


**Question:**

How does the data for claim videos and opinion videos compare or differ? Consider views, comments, likes, and shares.

Claim videos generally have higher engagement (views, likes, comments, and shares) than opinion videos, likely due to their more impactful or controversial content.


<img src="images/Construct.png" width="100" height="100" align=left>

## **PACE: Construct**

**Note**: The Construct stage does not apply to this workflow. The PACE framework can be adapted to fit the specific requirements of any project.




<img src="images/Execute.png" width="100" height="100" align=left>

## **PACE: Execute**

Consider the questions in your PACE Strategy Document and those below to craft your response.

### **Given your efforts, what can you summarize for Rosie Mae Bradshaw and the TikTok data team?**

*Note for Learners: Your answer should address TikTok's request for a summary that covers the following points:*

*   What percentage of the data is comprised of claims and what percentage is comprised of opinions?
*   What factors correlate with a video's claim status?
*   What factors correlate with a video's engagement level?


Summary for Rosie Mae Bradshaw and the TikTok Data Team:

Claim vs. Opinion: About 49.2% of the videos are claims, 48.5% are opinions, and 2.3% have missing claim status data.

Factors Correlating with Claim Status: Videos from banned or under review authors are more likely to be claims, and claim videos usually have higher views and shares.

Factors Correlating with Engagement: Claim videos get more engagement than opinion videos. Banned authors also see more likes and shares per view, indicating higher interaction.

In summary, claim videos, especially from banned authors, attract more engagement and views compared to opinion videos.

______________

Simplified for CEO and laypersons 

Glossary:

Claim Videos: Videos that present information or arguments, often trying to persuade the viewer or convey factual content.

Opinion Videos: Videos that express personal thoughts, views, or feelings, typically about a topic, without aiming to present factual evidence.

Claim Status: The classification of a video as either a claim or an opinion, indicating whether the video provides factual content or personal opinion.

Engagement: The level of interaction with a video, measured by likes, shares, comments, and views.

Author Ban Status: Indicates whether the video author is active, banned, or under review by the platform.

Likes Per View: A metric that measures the number of likes a video receives relative to the number of views.

Shares Per View: A metric that measures how often a video is shared relative to its number of views.

Comments Per View: A metric that measures how many comments a video receives relative to its number of views.

Banned Authors: Creators whose accounts have been suspended due to violating platform rules.

Under Review Authors: Creators whose content or account is being evaluated for possible policy violations.

Active Authors: Creators whose accounts are in good standing, without restrictions or suspensions.


**Congratulations!** You've completed this lab. However, you may not notice a green check mark next to this item on Coursera's platform. Please continue your progress regardless of the check mark. Just click on the "save" icon at the top of this notebook to ensure your work has been logged.
