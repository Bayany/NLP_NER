from glob import glob
import os
import re
# import nltk
# nltk.download('punkt')


files = glob("data/raw/PurpleHyacinth/*.txt")

for file in files:
    with open(file, 'r') as f:
        texts = f.readlines()[1:]
        new_texts = []
        for text in texts:
            # remove - ~ (used in splitted sentence between bubbles).
            t = re.sub(r"[_\-~\n]", ' ', text)
            # replace $ with s
            t = t.replace("$","s")
            # replace double ii in end of word with !!
            t = re.sub( r'\b(\w+)ii\b', r'\1i!', t)
            # remove numbers and special characters from begining and end of lines. (possibly mistook with drawing)
            cleaned_text = t.strip(" _-~123456789")
            # remove single character lines. (possibly mistook with drawing)
            if(len(cleaned_text) > 1):
                new_texts.append(cleaned_text)
            

        with open(file.replace("raw","clean"), 'w') as f2:
            f2.writelines(re.sub(r"\s+", " ", " ".join(new_texts)))





