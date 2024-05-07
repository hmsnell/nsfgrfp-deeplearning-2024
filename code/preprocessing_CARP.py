import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

filepath = '/Users/yzhu194/Documents/Brown/CSCI2470/FinalProject/nsfgrfp-deeplearning-2024/data/'
data = pd.read_csv(filepath+'pdf_texts.tsv', delimiter='\t')
data = data.drop(['title'], axis=1)
data['Success'] = data['Success'].replace({'Winner!': 0, 'HM': 1})
data = data.dropna(subset=['Success'])
data['Success'] = data['Success'].astype(int)
train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)
train_df.to_csv(filepath+'train.tsv',  sep='\t', index=False)
test_df.to_csv(filepath+'test.tsv',  sep='\t', index=False)
