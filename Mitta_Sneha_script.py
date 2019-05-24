import pandas as pd
import numpy as np
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
from gensim.models import Doc2Vec
from sklearn import utils
from sklearn.model_selection import train_test_split
import gensim
from sklearn.linear_model import LogisticRegression
from gensim.models.doc2vec import TaggedDocument
import re
import seaborn as sns
import matplotlib.pyplot as plt
import os
import glob
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
import multiprocessing
from gensim.test.test_doc2vec import ConcatenatedDoc2Vec

path = '/Users/snehamitta/Desktop/Fixed_Judgements/'

df1 = pd.DataFrame()

for infile in glob.glob( os.path.join(path, '*.*') ):
    fname = infile.replace("/", " ").split()[-1].split('.')[0]
    file = open(infile,"rb")
    data = file.read()
    # convert = data.split()
    # convert = preprocess(data)

    df1 = df1.append({'Judgements':fname, 'keywords':data}, ignore_index=True)

print(df1.head(5))

df2 = pd.read_csv("/Users/snehamitta/Desktop/Interview_Mapping.csv")
df1 = df1.merge(df2, how='left', on='Judgements')

df1.rename(columns = {'Area.of.Law':'LawType'}, inplace = True)
print(df1.head(5))

# cnt_pro = train['Area.of.Law'].value_counts()
# fig = plt.figure(figsize=(12,8))
# # train.groupby('Area.of.Law').keywords.count().plot.bar(ylim=0)
# sns.barplot(cnt_pro.index, cnt_pro.values, alpha=0.8)
# plt.ylabel('Number of Occurrences', fontsize=12)
# plt.xlabel('Area of Law', fontsize=12)
# plt.xticks(rotation=90)
# plt.tight_layout()
# plt.show()


def cleanText(text):
    text = BeautifulSoup(text, "lxml").text
    text = re.sub(r'\|\|\|', r' ', text) 
    text = re.sub(r'http\S+', r'<URL>', text)
    text = text.lower()
    text = text.replace('x', '')
    return text

df1['keywords'] = df1['keywords'].apply(cleanText)
print(df1.head(5))

s = 'To be Tested'

train = df1[df1['LawType'] != s]
test = df1[df1['LawType'] == s]

x_train = train['keywords']
y_train = train['LawType']

x_test = test['keywords']
y_test = test['LawType']


def tokenize_text(text):
    tokens = []
    for sent in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sent):
            if len(word) < 2:
                continue
            tokens.append(word.lower())
    return tokens

train_tagged = train.apply(
    lambda r: TaggedDocument(words=tokenize_text(r['keywords']), tags=[r.LawType]), axis=1)
test_tagged = test.apply(
    lambda r: TaggedDocument(words=tokenize_text(r['keywords']), tags=[r.LawType]), axis=1)

# print(train_tagged.values[30])

cores = multiprocessing.cpu_count()

model_dbow = Doc2Vec(dm=0, vector_size=300, negative=5, hs=0, min_count=2, sample = 0, workers=cores)
model_dbow.build_vocab([x for x in tqdm(train_tagged.values)])

for epoch in range(30):
    model_dbow.train(utils.shuffle([x for x in tqdm(train_tagged.values)]), total_examples=len(train_tagged.values), epochs=1)
    model_dbow.alpha -= 0.002
    model_dbow.min_alpha = model_dbow.alpha

def vec_for_learning(model, tagged_docs):
    sents = tagged_docs.values
    targets, regressors = zip(*[(doc.tags[0], model.infer_vector(doc.words, steps=20)) for doc in sents])
    return targets, regressors

def vec_for_learning(model, tagged_docs):
    sents = tagged_docs.values
    targets, regressors = zip(*[(doc.tags[0], model.infer_vector(doc.words, steps=20)) for doc in sents])
    return targets, regressors

y_train, x_train = vec_for_learning(model_dbow, train_tagged)
y_test, x_test = vec_for_learning(model_dbow, test_tagged)
logreg = LogisticRegression(n_jobs=1, C=1e5)
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_test)

print(y_pred)

model_dmm = Doc2Vec(dm=1, dm_mean=1, vector_size=300, window=10, negative=5, min_count=1, workers=5, alpha=0.065, min_alpha=0.065)
model_dmm.build_vocab([x for x in tqdm(train_tagged.values)])

for epoch in range(30):
    model_dmm.train(utils.shuffle([x for x in tqdm(train_tagged.values)]), total_examples=len(train_tagged.values), epochs=1)
    model_dmm.alpha -= 0.002
    model_dmm.min_alpha = model_dmm.alpha

y_train, x_train = vec_for_learning(model_dmm, train_tagged)
y_test, x_test = vec_for_learning(model_dmm, test_tagged)
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_test)

print(y_pred)

model_dbow.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
model_dmm.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)

new_model = ConcatenatedDoc2Vec([model_dbow, model_dmm])

def get_vectors(model, tagged_docs):
    sents = tagged_docs.values
    targets, regressors = zip(*[(doc.tags[0], model.infer_vector(doc.words, steps=20)) for doc in sents])
    return targets, regressors

y_train, x_train = get_vectors(new_model, train_tagged)
y_test, x_test = get_vectors(new_model, test_tagged)
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_test)

print(y_pred)
test['LawType'] = y_pred
test.drop(['keywords'], axis = 1, inplace = True)

# df_out = pd.merge(test,test[['LawType']],how = 'left',left_index = True, right_index = True)
print(test)
export = test.to_csv(r'/Users/snehamitta/Desktop/export_dataframe.csv', index = None, header=True)
