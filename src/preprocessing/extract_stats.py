
import pickle
from re import L
import pandas as pd
import numpy as np
import ast
from collections import defaultdict
import matplotlib.pyplot as plt
import dataframe_image as dfi

#  10 top words of LOC and PERSON
def ten_dissimilar_entities():

    with open("data/train_data.pickle", "rb") as f:
        labels_stats = {"non-unique": {"PERSON": 0, "LOC": 0},
                        "unique": {"PERSON": 0, "LOC": 0},
                        "sentence": {"PERSON": 0, "LOC": 0}}

        train_data = pickle.load(f)
        print("Length Train_data: ", len(train_data))
        labels_dict = {"PERSON": {}, "LOC": {}}

        for item in train_data:
            labels_flags = {"PERSON": False, "LOC": False}
            for wi, wj, label in item[1]:
                labels_flags[label] = True
                labels_stats["non-unique"][label] += 1
                key = item[0][wi:wj]
                labels_dict[label][key] = labels_dict[label].get(key, 0)+1
            for k, v in labels_flags.items():
                if(v):
                    labels_stats["sentence"][k] += 1

        top_10 = {}
        for key, value in labels_dict.items():
            labels_stats["unique"][key] = len(value)
            top_10[key] = list(
                dict(sorted(value.items(), key=lambda item: item[1], reverse=True)[:10]).keys())
        stat_dict = {'Tag': ['PERSON', 'LOC']}
        for i in range(1, 11):
            stat_dict["word"+str(i)] = [top_10["PERSON"]
                                        [i-1], top_10["LOC"][i-1]]

        pd.DataFrame(stat_dict).to_csv(
            "stats/ten_dissimilar_entities.csv", index=False)
    return labels_stats


#  TF-IDF
def calc_tf_idf():

    stat_dict = {'Tag': ['PERSON', 'LOC']}
    ki=0;
    for k in ["PERSON","LOC"]:
        words_df = pd.read_csv(f'data/wordbroken/{k}.csv')
        words_dict = defaultdict(int)
        for index, row in words_df.iterrows():
            row_words = ast.literal_eval(row["words"])
            for word in row_words:
                words_dict[word] += 1

        tf_idf = defaultdict(int)
        for index, row in words_df.iterrows():
            row_words = ast.literal_eval(row["words"])
            for idx, word in enumerate(row_words):
                tf = float(row_words.count(word)/len(row_words))
                idf = np.log(float(row_words.count(word))/words_dict[word])
                tf_idf[word] = round(tf*idf,2)
        tf_idf = sorted(tf_idf.items(), key=lambda x: x[1])
        for i in range(1, 11):
            if(ki==0):stat_dict["word"+str(i)]=["",""]
            stat_dict["word"+str(i)][ki] =  tf_idf[i-1]
        ki+=1
    stats_df=pd.DataFrame(stat_dict)
    stats_df.to_csv(
            "stats/ten_tf_idf_words.csv", index=False)
    dfi.export(stats_df,"stats/ten_tf_idf_words.png")



def calc_rnf():
    labels = ["PERSON", "LOC"]
    similar_words_dict = {}

    words_dict = {"PERSON": defaultdict(int),"LOC":  defaultdict(int)}
    for k in ["PERSON","LOC"]:
        words_df = pd.read_csv(f'data/wordbroken/{k}.csv')
        for index, row in words_df.iterrows():
            row_words = ast.literal_eval(row["words"])
            for word in row_words:
                words_dict[k][word] += 1
    rnf_df_cols=["word"+str(i) for i in range(1,11)]
    rnf_df = pd.DataFrame(columns=rnf_df_cols)
    for label1 in labels:
        for label2 in labels:
            if(label1==label2):continue
            similar_words = []
            for w in set(words_dict[label1]).intersection(set(words_dict[label2])):
                similar_words.append(w)
            similar_words_dict[f"{label1}/{label2}"] = similar_words
            similar_words_dict[f"{label2}/{label1}"] = similar_words
   
            rnf_dict = {}
            similar_words = similar_words_dict[f"{label1}/{label2}"]
            for word in similar_words:
                sorat =  words_dict[label1][word] / sum(words_dict[label1].values())
                makhraj = words_dict[label2][word] / sum(words_dict[label2].values())
                rnf_dict[word] = round((sorat / makhraj), 2)
            sorted_rnf = sorted(rnf_dict.items(), key=lambda x:x[1], reverse=True)
            for i in range(1,11):
                rnf_df.loc[f"{label1}/{label2}", "word"+str(i)] = sorted_rnf[i-1][0]
    rnf_df.to_csv("stats/relative_normalized_frequency.csv")


if(__name__ == "__main__"):

    labels=["Person","Loc"]
    labels_stats = ten_dissimilar_entities()

    # #  sentence counts
    sentences_df = pd.read_csv(f'data/sentencebroken/all_chapters.csv')

    sentences_count = {"Total": len(sentences_df),
                       "PERSON": [labels_stats["sentence"]["PERSON"]],
                       "LOC": [labels_stats["sentence"]["LOC"]]}
    pd.DataFrame(sentences_count).to_csv(
        'stats/sentences_count.csv', index=False)

    #  unique word counts
    words_df = pd.read_csv(f'data/wordbroken/all_chapters.csv')
    words_dict = defaultdict(int)
    total_words = 0
    for index, row in words_df.iterrows():
        row_words = ast.literal_eval(row["words"])
        total_words += len(row_words)
        for word in row_words:
            if(len(word) > 1 and not "'" in word):
                words_dict[word] += 1

    words_dict = sorted(words_dict.items(), key=lambda x: x[1], reverse=1)
    unique_words_count = {"Total": [len(words_dict)],
                          "PERSON": [labels_stats["unique"]["PERSON"]],
                          "LOC": [labels_stats["unique"]["LOC"]]}
    pd.DataFrame(unique_words_count).to_csv(
        'stats/unique_words_count.csv', index=False)

    #  words countss
    words_count = {"Total": [total_words],
                   "PERSON": [labels_stats["non-unique"]["PERSON"]],
                   "LOC": [labels_stats["non-unique"]["LOC"]]}
    pd.DataFrame(words_count).to_csv('stats/words_count.csv', index=False)

    calc_tf_idf()

    calc_rnf()

    # Histogram
    hist_show = dict(words_dict[:15])
    plt.bar(list(hist_show.keys()), hist_show.values(), color='purple')
    plt.savefig('stats/hist.png')
    plt.show()
