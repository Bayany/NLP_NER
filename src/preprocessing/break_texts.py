import spacy
from glob import glob
import nltk
from nltk.tokenize import word_tokenize
from pandas import DataFrame as df
import pickle
nltk.download('punkt')

nlp = spacy.load("en_core_web_sm")


def detect_sentences(text):
    doc = nlp(text)
    sentences = []
    current_sentence = []

    for token in doc:
        current_sentence.append(token.text)

        if token.is_sent_start or token.text in ['?', '!']:
            sentences.append(' '.join(current_sentence))
            current_sentence = []

    if current_sentence:
        sentences.append(' '.join(current_sentence))

    return sentences


if __name__ == "__main__":
    files = glob("data/clean/PurpleHyacinth/*.txt")
    words_info = []
    sent_info = []

    for file in files:
        with open(file, 'r') as f:
            text = "".join(f.readlines())
            chap_no = file.split("ch")[-1][:-4]
            words = word_tokenize(text)
            sentences = detect_sentences(text)
            words_info.append({"chapter_no": chap_no, "words": words})
            for sent in sentences:
                sent_info.append({"chapter_no": chap_no, "sentence": sent})

    words_df = df(words_info, columns=['chapter_no', 'words'])
    words_df.to_csv("data/wordbroken/all_chapters.csv", index=False)
    sent_df = df(sent_info, columns=['chapter_no', 'sentence'])
    sent_df.to_csv("data/sentencebroken/all_chapters.csv", index=False)

    with open("data/train_data.pickle", "rb") as f:

        train_data = pickle.load(f)
        sent_info.append({"chapter_no": chap_no, "sentence": sent})
        labels_dict = {"PERSON": {"sentence": [], "word": []},
                       "LOC": {"sentence": [], "word": []}}
        for id, item in enumerate(train_data):
            labels_flags = {"PERSON": False, "LOC": False}
            for wi, wj, label in item[1]:
                labels_flags[label] = True
            for k, v in labels_flags.items():
                if(v):
                    labels_dict[k]["sentence"].append({"id": id, "sentence": item[0]})
                    labels_dict[k]["word"].append({"id": id, "words": word_tokenize(item[0])})

        for k in labels_flags.keys():
            df(labels_dict[k]["word"], columns=['id', 'words']).to_csv(
                f"data/wordbroken/{k}.csv", index=False)
            df(labels_dict[k]["sentence"], columns=['id', 'sentence']).to_csv(
                f"data/sentencebroken/{k}.csv", index=False)
