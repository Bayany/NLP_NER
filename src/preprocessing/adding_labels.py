import pandas as pd
import pickle
import json

labels_df = pd.read_csv('data/clean/labels.csv')
texts_df = pd.read_csv('data/sentencebroken/all_chapters.csv')

train_data = []

lookup_tb={}

for index, row in labels_df.iterrows():
    print(row)
    label = row['Label'] 
    word = row['Entity']
    print("----")
    for text_id, row2 in texts_df.iterrows():
        text=row2['sentence']
        wi=text.lower().find(word.lower())
        if(wi==-1):continue
        if(not text_id in lookup_tb):
            lookup_tb[text_id]=len(train_data)
            train_data.append((text, []))
        train_data[lookup_tb[text_id]][1].append((wi,wi+len(word),label))




  
with open('data/train_data.pickle', 'wb') as f:
    pickle.dump(train_data, f)

with open("data/clean/train_data.json", 'w') as f:
    json.dump(train_data, f)