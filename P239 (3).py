#!/usr/bin/env python
# coding: utf-8

# # Twitter Semantic Analysis:

# ## Import the Necessary Dependencies

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
import re


# In[3]:


# Input the data
data=pd.read_csv('C:/Users/Pooja Patil/Downloads/tweet.csv',encoding='latin-1')


# In[4]:


data


# # Exploratory Data Analysis

# In[5]:


# Five top records of data
data.head()


# In[6]:


# Columns/features in data
data.columns


# In[7]:


## Length of the dataset
print('length of data is', len(data))


# In[8]:


# Shape of data
data.shape


# In[9]:


## Data information
data.info()


# In[10]:


# Datatypes of all columns
data.dtypes


# In[11]:


#Checking for null values
np.sum(data.isnull().any(axis=1))


# In[12]:


# Rows and columns in the dataset
print('Count of columns in the data is:  ', len(data.columns))
print('Count of rows in the data is:  ', len(data))


# In[13]:


# Check unique class values
data['class'].unique()


# # Data Visualization of class Variables

# In[14]:


#Plotting the distribution for dataset.
ax = data.groupby('class').count().plot(kind='bar', title='Distribution of data',legend=False)
ax.set_xticklabels(['figurative', 'irony', 'regular', 'sarcasm'], rotation=0)
# Storing data in lists.
tweets, sentiment = list(data['tweets']), list(data['class'])


# In[15]:


import seaborn as sns
sns.countplot(x='class', data=data)


# In[16]:


class_counts=data['class'].value_counts()
class_counts


# In[17]:


plt.pie(class_counts, labels=class_counts.index,autopct='%1.1f%%')
plt.title('Class Distribution')
plt.show()


# # Data Cleaning and Preprocessing

# In[18]:


import re
s="string .with .Punctuation?"
s=re.sub(r'[^\w\s]','',s)


# In[19]:


import string


# ## Removal of punctuations

# In[20]:


string.punctuation


# In[21]:


def remove_punctuations(text):
    punctuations=string.punctuation
    return text.translate(str.maketrans('','',punctuations))


# In[22]:


data['new_tweets']=data['tweets'].apply(lambda x: remove_punctuations(x))


# In[23]:


data.head()


# ## Removal of stopwords

# In[24]:


from nltk.corpus import stopwords
stopwords.words('english')


# In[109]:


" , ".join(stopwords.words('english'))


# In[113]:


STOPWORDS=set(stopwords.words('english'))
def removal_stopwards(text):
    return" ".join([word for word in text.split() if word not in STOPWORDS])


# In[114]:


data['new_tweets']=data['new_tweets'].apply(lambda x: removal_stopwards(x))
data.head()


# ## Removal of Frequent word

# In[25]:


from collections import Counter
word_count=Counter()
for text in data['new_tweets']:
    for word in text.split():
        word_count[word] +=1
            
word_count.most_common(10)            


# In[26]:


word_count.most_common()            


# ## Ferquent word

# In[27]:


FREQUENT_WORDS=set(word for (word, wc)in word_count.most_common(3))
def removal_freq_words(text):
    return" ".join([word for word in text.split() if word not in FREQUENT_WORDS])


# In[28]:


data['new_tweets']=data['new_tweets'].apply(lambda x: removal_freq_words(x))
data.head()


# ## Removal of Rare Words

# In[29]:


RARE_WORDS=set(word for (word, wc)in word_count.most_common()[:-10:-1])
RARE_WORDS


# ## Removing URLs

# In[30]:


import re
def remove_url(text):
    return re.sub('((www.[^s]+)|(https?//[^s]+))|(https?://[^s]+)', '',text)

data['new_tweets']=data['new_tweets'].apply(remove_url)
data.head(10)


# In[31]:


def removal_rare_words(text):
    return" ".join([word for word in text.split() if word not in RARE_WORDS])


# In[32]:


data['new_tweets']=data['new_tweets'].apply(lambda x: removal_rare_words(x))
data.head()


# # Removal of Special Characters

# In[33]:


import re
def removal_spl_chars(text):
    text=re.sub('[^a-zA-Z0-9]',' ',text)
    text=re.sub('\s+',' ',text)
    return text


# In[34]:


data['new_tweets']=data['new_tweets'].apply(lambda x: removal_spl_chars(x))
data.head()


# # Stemming

# In[35]:


from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
def stem_words(text):
    return " ".join([ps.stem(word) for word in text.split()])


# In[36]:


data['Stemmed_tweets']=data['new_tweets'].apply(lambda x: stem_words(x))
data.head()


# # Lemmatization

# In[37]:


lm=nltk.WordNetLemmatizer()
def lemmatizer_on_text(data):
    text=[st.stem(word)for word in data]
    return data


# In[38]:


data['lemmatized_tweets']=data['new_tweets'].apply(lambda x: stem_words(x))
data.head()


# # Tokenization 

# In[39]:


from nltk.tokenize import word_tokenize
data['clean_tweets']=data['new_tweets'].astype(str)
data['clean_tweets']=data['new_tweets'].apply(nltk.word_tokenize)
data


# # Separating input features and Target

# In[40]:


x=data['new_tweets']
y=data['class']


# In[41]:


x


# In[42]:


y


# ## Plotting a cloud of words for figrative

# In[43]:


tweets=" ".join(data['new_tweets'])
type(tweets)


# In[44]:


from wordcloud import WordCloud
wordcloud_stw= WordCloud(
               background_color ='black',
               width = 1800,
               height = 1500
               ).generate(tweets)
plt.imshow(wordcloud_stw)
plt.title('WordCloud for Tweets Data')
plt.show()


# ### Understanding the common words used in the tweets: WordCloud

# In[45]:


from wordcloud import WordCloud 
wordcloud=WordCloud(width=800, height=500,random_state=21,
1ds                    max_font_size=110).generate(tweets)

plt.figure(figsize=(10,7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()


# In[ ]:


# visualize the frequent words
all_words = " ".join([sentence for sentence in data['new_tweets']])

from wordcloud import WordCloud
wordcloud = WordCloud(width=800, height=500, random_state=42, max_font_size=100).generate(all_words)

# plot the graph
plt.figure(figsize=(15,8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# ### Top 10 most frequent words

# In[ ]:


from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Combine all tweets into a single string
all_tweets = ' '.join(data['new_tweets'])

# Split the string into individual words
words = all_tweets.split()

# Count the frequency of each word
word_freq = Counter(words)

# Get the most common words and their frequencies
top_words = word_freq.most_common(10)

# Create a bar chart of the top words
x = [word[0] for word in top_words]
y = [count[1] for count in top_words]

plt.bar(x, y)
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.title('Top 10 Most Frequent Words')
plt.xticks(rotation=45)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




