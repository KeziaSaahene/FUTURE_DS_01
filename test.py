import pandas as pd
import numpy as np
from textblob import TextBlob


df= pd.read_csv('sentimentdataset.csv')

# copying csv file
sentiment_df= df.copy()


# NLP
def cat_sent(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity

    if polarity > 0:
        return 'Positive'
    elif polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'

sentiment_df['sentiment_class']= sentiment_df['Sentiment'].apply(cat_sent)
print(sentiment_df['sentiment_class'])

sentiment_df.to_csv('sentiment_df.csv', index = False)

print(sentiment_df.head())