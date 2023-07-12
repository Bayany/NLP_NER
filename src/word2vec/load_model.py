import pandas as pd
import numpy as np
from treebank import *
from sgd import load_saved_params
import dataframe_image as dfi


if __name__ == '__main__':

    labels_dict = {'PERSON': set(), 'LOC': set()}
    datasets = {}
    params={}
    for label in ['PERSON', 'LOC']:
        datasets[label] = WebtoonSentiment(f'data/sentencebroken/{label}.csv')
        start_iter, params[label], state = load_saved_params(label)
        for sent in datasets[label].sentences():
            for word in sent:
                if(len(word) > 3):
                    labels_dict[label].add(word)
    similar_words = labels_dict["PERSON"].intersection(labels_dict["LOC"])
    report = []
    for word in similar_words:
        wi1 = datasets["PERSON"].tokens()[word]
        wi2 = datasets["LOC"].tokens()[word]
        x = params["PERSON"][wi1]
        y = params["LOC"][wi2]
        cosine_similarity=round(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)),2)
        report.append([word, cosine_similarity])
    report_df = pd.DataFrame(report, columns=['word', 'cosine similarity'])
    report_df.to_csv(
        'reports/word2vec_cosine_similarity.csv', index=False)
    dfi.export(report_df[:20],"reports/word2vec_cosine_similarity.png")