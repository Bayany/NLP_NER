import nltk
from collections import defaultdict
import matplotlib.pyplot as plt
from glob import glob

data_path = "data/clean/PurpleHyacinth/*.txt"

nltk.download('punkt')

files = glob(data_path)
count_line = 0

words_count = 0
words_dict = defaultdict(int)

for file in files:
    with open(file, 'r') as f:
        for text in f:
            text = f.read()
            count_line += 1
            words = nltk.word_tokenize(text)
            words_count += len(words)

            for word in words:
                words_dict[word] += 1

words_dict = sorted(words_dict.items(), key=lambda x: x[1], reverse=1)

print(count_line)
print(words_count)



hist_show = dict(words_dict[:15])
plt.bar(list(hist_show.keys()), hist_show.values(), color='g')
plt.savefig(f'hist.png')
plt.show()