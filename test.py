from pyvi import ViTokenizer, ViPosTagger
import re
import numpy as np
import collections
def build_dataset(words, n_words):
    count = [['GO', 0], ['PAD', 1], ['EOS', 2], ['UNK', 3]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        index = dictionary.get(word, 0)
        if index == 0:
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary


file_path = './data/'
with open(file_path+'test.txt', 'r', encoding='utf-8') as fopen:
    text_test = fopen.read().lower()
    text_test = re.sub('[“”!@#$\n]', '', text_test)
    text_test = text_test.split('.')
concat_test =[]
for i in text_test:
    k= ViTokenizer.tokenize(i).split(' ')
    concat_test = np.append(concat_test,k)
print(concat_test)
vocabulary_size_from = len(list(set(concat_test)))
data_test, count_test, dictionary_test, rev_dictionary_test = build_dataset(concat_test, vocabulary_size_from)
print (dictionary_test)
def str_idx(corpus, dic):
    X = []
    for i in corpus:
        ints = []
        m = ViTokenizer.tokenize(i).split(' ')
        for k in m:
            try:
                ints.append(dic[k])
            except Exception as e:
                print(e)
                ints.append(2)
        X.append(ints)
    return X
X = str_idx(text_test, dictionary_test)
print (X)