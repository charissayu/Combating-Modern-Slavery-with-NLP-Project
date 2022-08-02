#!/usr/bin/env python
# coding: utf-8

# In[118]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import torch
import torch.nn as nn

import transformers
from transformers import BertTokenizer, BertForSequenceClassification



GT = pd.read_csv('GT_MSA_Whistleblowing_VerifiedDataLabels_WithText.csv')
GT.rename(columns={'TEXT':'Text'}, inplace=True)

GT_new = pd.read_csv('Ground_Truth_MSA_Whistleblowing_NewDataLabels_WithText.csv')


df = pd.concat([GT, GT_new],axis=0)


df_1 = df.dropna(subset=['Text'])

df_1 = df_1[['Company', 'Value', 'Text', 'Comments']]
df_1.reset_index(inplace=True, drop=True)


df_0906 = pd.read_csv('0906.csv')
df_0906 = df_0906[['Value', 'Text', 'label']]
df_0906

df_1['label'] = df_0906.label


# remove rows 'Text' column does not contains the key sentences 

no_text =     [30, 35, 36, 152, 203, 208, 360, 391, 398, 415, 424, 429, 442, 444, 447, 588, 591, 604, 608, 614, 
              620, 626, 653, 657, 659, 684, 691, 844, 849, 865, 875, 899, 911, 905, 1186]
delete_row =  [1069, 158, 442]
wrong_label = [402, 905]

delete_rows = no_text + delete_row + wrong_label
len(delete_rows)


df_1.drop(delete_rows, inplace=True)
df_1


# In[127]:


df_1.reset_index(inplace=True, drop=True)
df_1


# In[129]:


#df_1.to_csv('df_1018.csv')


# In[164]:


df_1018 = pd.read_csv('df_1018.csv')
df_1018


# In[167]:


df_1018.label.value_counts()


# ### Text pre-processing 
# 1. Break down document to sentence corpus 
# 2. stop word removal & stemming

# In[169]:


import nltk 
from nltk.stem import PorterStemmer 
from nltk.tokenize import word_tokenize
ps = PorterStemmer()
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))


sentence_corpus_1_1018 = []
for i in range(df_1018.shape[0]):
    lower_text = df_1018.iloc[i]['Text'].lower()
    text = nltk.tokenize.sent_tokenize(lower_text)
    output = []
    for sentence in text:
        output.append(" ".join(sentence.split()))
    sentence_corpus_1_1018.append(output)


# In[170]:


len(sentence_corpus_1_1018)


# In[171]:


length_list_1 = []
long_doc_count_1 = 0
short_doc_count_1 = 0
short_doc_idx_1 = []
long_doc_idx_1 = []


sentence_corpus_1 = sentence_corpus_1_1018
for i in range(len(sentence_corpus_1)):
    length = len(sentence_corpus_1[i])
    length_list_1.append(length)
    
    if length > 200:
        long_doc_count_1 +=1
        long_doc_idx_1.append(i)
        
    elif length < 5:
        short_doc_count_1 +=1
        short_doc_idx_1.append(i)
    
print('short_doc_count_1', short_doc_count_1)
print('long_doc_count_1', long_doc_count_1)


# In[172]:


def contains_word(sentence, keyword_list):
    sent_key_list = []
    for keyword in keyword_list:
        if keyword in sentence:
            sent_key_list.append(1)
        else:
            sent_key_list.append(0)
    if 1 in sent_key_list:
        sent_key = 1
    else: 
        sent_key = 0
    return sent_key 


# In[173]:


from re import search

df_csv = df_1018

wrong_label = []
correct_label = []
for i in range(len(df_csv)):
    if (search('No|no', df_csv['Value'][i])) and  (df_csv['label'][i] ==1):
        wrong_label.append(i)
    elif (search('Development|development', df_csv['Value'][i])) and (df_csv['label'][i] ==1):
        wrong_label.append(i)
    elif (search('No|no', df_csv['Value'][i])) and  (df_csv['label'][i] ==0):
        correct_label.append(i)
    elif (search('Development|development', df_csv['Value'][i])) and (df_csv['label'][i] ==0):
        correct_label.append(i)
    else:
        continue
        
wrong_label


# ### Text pre-processing 
# - stop word removal & Stemming
# - Break down document to sentence corpus

# In[3]:


#sentence_corpus_2_1018   - stemmed sentence document
#sentence_corpus_3_1018   - stemmed sentence with keyword only
#sentence_corpus_3a_1018  - stemmed sentence with keyword >= 2
#sentence_corpus_3b_1018  - stemmed sentence with keyword >=2 and 'NA' for lines without keyword
#sentence_corpus_3c_1018  - standardized hotline, whole sentence corpus
#sentence_corpus_3e_1018  - standardized whistleblow, whole sentence corpus
#sentence_corpus_3g_1018  - keywords only
#sentence_corpus_3g1_1018 - keywords only, empty list to 'NA'


# In[174]:


sentence_corpus_2_1018 = []
for i in range(len(sentence_corpus_1_1018)):
    doc_sentence = []
    for r in range(len(sentence_corpus_1_1018[i])):
        word_tokenize_sentence = word_tokenize(sentence_corpus_1_1018[i][r])
        stem_sentence = []
        for w in word_tokenize_sentence:
            if w not in stop_words:
                stem_word = ps.stem(w)
                stem_sentence.append(stem_word)
        sentence = ' '.join(stem_sentence)
        doc_sentence.append(sentence)
    sentence_corpus_2_1018.append(doc_sentence)
        


# In[175]:


sentence_corpus_2_1018


# In[327]:


all_sentence_length = []
for i in range(len(sentence_corpus_2_1018)):
    sentence_length = len(sentence_corpus_2_1018[i])
    all_sentence_length.append(sentence_length)
    


# In[332]:


plt.figure(figsize=(9,6.5))
pd.Series(all_sentence_length).hist(bins=100, color='slategray' )

plt.xlabel('No.of sentences')
plt.ylabel('documents')
plt.title('Distribution of document length')


# In[182]:


#stemmed sentence with keyword only

sentence_corpus_3_1018 = []
for i in range(len(sentence_corpus_2_1018)):
    sentence_corpus_doc = []
    for r in range(len(sentence_corpus_2_1018[i])):
        for k in range(len(keyword_list_1018)):
            if ((keyword_list_1018[k] in sentence_corpus_2_1018[i][r]) and (sentence_corpus_2_1018[i][r] not in sentence_corpus_doc)):
                sentence_corpus_doc.append(sentence_corpus_2_1018[i][r])
    sentence_corpus_3_1018.append(sentence_corpus_doc)
    


# In[183]:


#stemmed sentence with keyword >= 2

sentence_corpus_3a_1018 = []
for i in range(len(sentence_corpus_3_1018)):
    doc_sentence = []
    for r in range(len(sentence_corpus_3_1018[i])):
        sentence_keyword_count = []
        for keyword in keyword_list_1018:
            if keyword in sentence_corpus_3_1018[i][r]:
                sentence_keyword_count.append(keyword)
        if len(sentence_keyword_count) >= 2:
            print(len(sentence_keyword_count))
            print(sentence_corpus_3_1018[i][r])
            doc_sentence.append(sentence_corpus_3_1018[i][r])
        else:
            continue
    sentence_corpus_3a_1018.append(doc_sentence)
    


# In[184]:


#stemmed sentence with keyword >=2 and 'NA' for lines without keyword

sentence_corpus_3b_1018 = []
for i in range(len(sentence_corpus_3a_1018)):
    if len(sentence_corpus_3a_1018[i])==0:
        sentence_corpus_3b_1018.append('NA')
    elif len(sentence_corpus_3a_1018[i]) > 0:
        join_text = ' '.join(sentence_corpus_3a_1018[i])
        sentence_corpus_3b_1018.append(join_text)


# In[185]:


#standardized hotline, whole sentence corpus

sentence_corpus_3c_1018 = []

for i in range(len(sentence_corpus_3b_1018)):
    if 'helplin' in sentence_corpus_3b_1018[i]:
        x = sentence_corpus_3b_1018[i].replace('helplin', 'hotlin')
        sentence_corpus_3c_1018.append(x)
        
    elif 'help-lin' in sentence_corpus_3b_1018[i]:
        x = sentence_corpus_3b_1018[i].replace('help-lin', 'hotlin')
        sentence_corpus_3c_1018.append(x)
        
    elif 'help line' in sentence_corpus_3b_1018[i]:
        x = sentence_corpus_3b_1018[i].replace('help line', 'hotlin')
        sentence_corpus_3c_1018.append(x)
        
    elif 'line' in sentence_corpus_3b_1018[i]:
        x = sentence_corpus_3b_1018[i].replace('line', 'hotlin')
        sentence_corpus_3c_1018.append(x)
    
    elif 'telephon' in sentence_corpus_3b_1018[i]:
        x = sentence_corpus_3b_1018[i].replace('telephon', 'hotlin')
        sentence_corpus_3c_1018.append(x)
        
    elif 'phone' in sentence_corpus_3b_1018[i]:
        x = sentence_corpus_3b_1018[i].replace('phone', 'hotlin')
        sentence_corpus_3c_1018.append(x)
        
    elif 'number' in sentence_corpus_3b_1018[i]:
        x = sentence_corpus_3b_1018[i].replace('number', 'hotlin')
        sentence_corpus_3c_1018.append(x)
    else:
         sentence_corpus_3c_1018.append(sentence_corpus_3b_1018[i])


# In[186]:


#standardized whistleblow, whole sentence corpus

sentence_corpus_3e_1018 = []

for i in range(len(sentence_corpus_3c_1018)):
    if 'whistle-blow' in sentence_corpus_3c_1018[i]:
        x = sentence_corpus_3c_1018[i].replace('whistle-blow', 'whistleblow')
        sentence_corpus_3e_1018.append(x)
        
    elif 'whistl blow' in sentence_corpus_3c_1018[i]:
        x = sentence_corpus_3c_1018[i].replace('whistl blow', 'whistleblow')
        sentence_corpus_3e_1018.append(x)
        
    elif 'whistl blower' in sentence_corpus_3c_1018[i]:
        x = sentence_corpus_3c_1018[i].replace('whistl blower', 'whistleblow')
        sentence_corpus_3e_1018.append(x)
    else:
         sentence_corpus_3e_1018.append(sentence_corpus_3c_1018[i])


# In[187]:


len(sentence_corpus_3e_1018)


# In[2]:


stem_keywords_1 = ['whistleblow','independ','incid','report','bring ani concern','rais','concern','suspect incid',
 'bring issu','encourag','complianc','ethic','hotlin','helplin', 'help-lin','help line','line','telephon',
 'phone','number','line','email','channel','onlin portal','grievanc','mechan','mailto','speak up','24/7','24-hour',
 '24 hour','ethicspoint','board','member','manag','depart','team','hr','senior','offic','director','ombudsman',
 'integr','head','human resourc','risk','audit committe','contact','ethic','supervisor','legal','depart','risk',
 'compani secretari','team','attent','workplac repres','qualiti','execut','nomin','address','focal point',
 'anonym','confidenti','complianc','in confid','detriment','treatment','without fear','repris','protect',
 'against','without ani risk','freedom','discrimin','retali','retribut','victimis','recrimin','anti-retali',
 'dismiss','disciplinari action','threat','unfair','repercuss','non-retali']

print(len(stem_keywords_1))


# In[190]:


#keywords only, list of words not string
sentence_corpus_3h_1018 = []
for i in range(len(sentence_corpus_3e_1018)):
    sentence_corpus_3g = []
    for keyword in stem_keywords_1:
        if keyword in sentence_corpus_3e_1018[i]:
            sentence_corpus_3g.append(keyword)
    sentence_corpus_3h_1018.append(sentence_corpus_3g)


# In[191]:


#keywords only corpus
sentence_corpus_3g_1018 = []
for i in range(len(sentence_corpus_3h_1018)):
    text = ' '.join(sentence_corpus_3h_1018[i])
    sentence_corpus_3g_1018.append(text)
    
sentence_corpus_3g_1018 


# In[192]:


#keywords only corpus, change empty string to 'NA'
sentence_corpus_3g1_1018 = []
for i in range(len(sentence_corpus_3g_1018)):
    if len(sentence_corpus_3g_1018[i])> 0:
        sentence_corpus_3g1_1018.append(sentence_corpus_3g_1018[i])
    elif len(sentence_corpus_3g_1018[i])== 0:
        sentence_corpus_3g1_1018.append('NA')


# In[193]:


sentence_corpus_3g1_1018


# In[194]:


ky1_1 = []
ky1_0 = []
ky0_1 = []
ky0_0 = []
for i in range(len(sentence_corpus_3g_1018)):
    if len(sentence_corpus_3g_1018[i])>0 and df_1018.label[i] == 1:
        ky1_1.append(i)
    elif len(sentence_corpus_3g_1018[i])>0 and df_1018.label[i] == 0:
        ky1_0.append(i)
    elif len(sentence_corpus_3g_1018[i])==0 and df_1018.label[i] == 0:
        ky0_0.append(i)
    elif len(sentence_corpus_3g_1018[i])==0 and df_1018.label[i] == 1:
        ky0_1.append(i)
    
print(len(ky1_1))
print(len(ky1_0))
print(len(ky0_1))
print(len(ky0_0))


# ### TF-IDF - sentence representation

# In[196]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[197]:


tf_idf_1 = TfidfVectorizer(ngram_range=(1, 1),binary=True,smooth_idf=False)
tf_idf_2 = TfidfVectorizer(ngram_range=(2, 2),binary=True,smooth_idf=False)
tf_idf_3 = TfidfVectorizer(ngram_range=(1, 2),binary=True,smooth_idf=False)

w_corpus = sentence_corpus_3e_1018 #whole sentence corpus

data_0_t = pd.DataFrame(tf_idf_1.fit_transform(w_corpus).todense(), columns= tf_idf_1.vocabulary_)
data_0_t


# In[55]:


data_1_t = pd.DataFrame(tf_idf_2.fit_transform(w_corpus).todense(), columns= tf_idf_2.vocabulary_)
data_1_t


# In[56]:


data_2_t = pd.DataFrame(tf_idf_3.fit_transform(w_corpus).todense(), columns= tf_idf_3.vocabulary_)
data_2_t


# In[202]:


#keywords only corpus
k_corpus = sentence_corpus_3g_1018 

data_3_t = pd.DataFrame(tf_idf_1.fit_transform(k_corpus).todense(), columns= tf_idf_1.vocabulary_)
data_3_t


# In[58]:


data_4_t = pd.DataFrame(tf_idf_2.fit_transform(k_corpus).todense(), columns= tf_idf_2.vocabulary_)
data_4_t


# In[59]:


data_5_t = pd.DataFrame(tf_idf_3.fit_transform(k_corpus).todense(), columns= tf_idf_3.vocabulary_)
data_5_t


# ### Model training - SVC

# In[61]:


from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

#whole_unigram
Result = []
train_result = []
for i in range(0,10):
    X_train, X_test, y_train, y_test = train_test_split(data_0_t, df_1018.label, test_size=0.2, random_state=i)

    clf = SVC(C=1.7)
    clf.fit(X_train, y_train)
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    print(i, 'train_acc:', round(train_acc, 3), 'test_acc:', round(test_acc, 3))
    
    Result.append(test_acc)
    train_result.append(train_acc)
    
print('average train acc:', round(sum(train_result)/len(train_result),3))
print('average test acc:', round(sum(Result)/len(Result),3))
print('best acc:', 'test acc', round(max(Result), 3), 'train acc', round(max(train_result), 3))


# In[62]:


#whole_bigram
Result = []
train_result = []
for i in range(0,10):
    X_train, X_test, y_train, y_test = train_test_split(data_1_t, df_1018.label, test_size=0.2, random_state=i)

    clf = SVC(C=1.6)
    clf.fit(X_train, y_train)
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    print(i, 'train_acc:', round(train_acc, 3), 'test_acc:', round(test_acc, 3))
    
    Result.append(test_acc)
    train_result.append(train_acc)
    
print('average train acc:', round(sum(train_result)/len(train_result),3))
print('average test acc:', round(sum(Result)/len(Result),3))
print('best acc:', 'test acc', round(max(Result), 3), 'train acc', round(max(train_result), 3))


# In[63]:


#whole_uni+bigram
Result = []
train_result = []
for i in range(0,10):
    X_train, X_test, y_train, y_test = train_test_split(data_2_t, df_1018.label, test_size=0.2, random_state=i)

    clf = SVC(C=1.6)
    clf.fit(X_train, y_train)
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    print(i, 'train_acc:', round(train_acc, 3), 'test_acc:', round(test_acc, 3))
    
    Result.append(test_acc)
    train_result.append(train_acc)
    
print('average train acc:', round(sum(train_result)/len(train_result),3))
print('average test acc:', round(sum(Result)/len(Result),3))
print('best acc:', 'test acc', round(max(Result), 3), 'train acc', round(max(train_result), 3))


# In[64]:


#key_unigram
Result = []
train_result = []
for i in range(0,10):
    X_train, X_test, y_train, y_test = train_test_split(data_3_t, df_1018.label, test_size=0.2, random_state=i)

    clf = SVC(C=1.7)
    clf.fit(X_train, y_train)
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    print(i, 'train_acc:', round(train_acc, 3), 'test_acc:', round(test_acc, 3))
    
    Result.append(test_acc)
    train_result.append(train_acc)
    
print('average train acc:', round(sum(train_result)/len(train_result),3))
print('average test acc:', round(sum(Result)/len(Result),3))
print('best acc:', 'test acc', round(max(Result), 3), 'train acc', round(max(train_result), 3))


# In[65]:


#key_bigram
Result = []
train_result = []
for i in range(0,10):
    X_train, X_test, y_train, y_test = train_test_split(data_4_t, df_1018.label, test_size=0.2, random_state=i)

    clf = SVC(C=1.7)
    clf.fit(X_train, y_train)
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    print(i, 'train_acc:', round(train_acc, 3), 'test_acc:', round(test_acc, 3))
    
    Result.append(test_acc)
    train_result.append(train_acc)
    
print('average train acc:', round(sum(train_result)/len(train_result),3))
print('average test acc:', round(sum(Result)/len(Result),3))
print('best acc:', 'test acc', round(max(Result), 3), 'train acc', round(max(train_result), 3))


# In[66]:


#key_uni+bigram
Result = []
train_result = []
for i in range(0,10):
    X_train, X_test, y_train, y_test = train_test_split(data_5_t, df_1018.label, test_size=0.2, random_state=i)

    clf = SVC(C=1.7)
    clf.fit(X_train, y_train)
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    print(i, 'train_acc:', round(train_acc, 3), 'test_acc:', round(test_acc, 3))
    
    Result.append(test_acc)
    train_result.append(train_acc)
    
print('average train acc:', round(sum(train_result)/len(train_result),3))
print('average test acc:', round(sum(Result)/len(Result),3))
print('best acc:', 'test acc', round(max(Result), 3), 'train acc', round(max(train_result), 3))


# ### Model training - MLP

# In[83]:


from sklearn.neural_network import MLPClassifier

#whole_unigram

Result = []
train_result = []
for i in range(0,10):
    X_train, X_test, y_train, y_test = train_test_split(data_0_t, df_1018.label, test_size=0.2, random_state=i)
    clf = MLPClassifier(hidden_layer_sizes=(10000,500), random_state=i, max_iter=200,  alpha = 0.001, early_stopping=True).fit(X_train, y_train)
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    print(i , 'train_acc:', round(train_acc,4), 'test_acc:', round(test_acc, 4))
    
    Result.append(test_acc)
    train_result.append(train_acc)
    
print('average train acc:', round(sum(train_result)/len(train_result),3))
print('average test acc:', round(sum(Result)/len(Result),3))
print('best acc:', 'test acc', round(max(Result), 3), 'train acc', round(max(train_result), 3))


# In[84]:


#whole_bigram

Result = []
train_result = []
for i in range(0,10):
    X_train, X_test, y_train, y_test = train_test_split(data_1_t, df_1018.label, test_size=0.2, random_state=i)
    clf = MLPClassifier(hidden_layer_sizes=(10000,500), random_state=i, max_iter=200,  alpha = 0.001, early_stopping=True).fit(X_train, y_train)
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    print(i , 'train_acc:', round(train_acc,4), 'test_acc:', round(test_acc, 4))
    
    Result.append(test_acc)
    train_result.append(train_acc)
    
print('average train acc:', round(sum(train_result)/len(train_result),3))
print('average test acc:', round(sum(Result)/len(Result),3))
print('best acc:', 'test acc', round(max(Result), 3), 'train acc', round(max(train_result), 3))


# In[85]:


#whole_unigram+bigram

Result = []
train_result = []
for i in range(0,10):
    X_train, X_test, y_train, y_test = train_test_split(data_2_t, df_1018.label, test_size=0.2, random_state=i)
    clf = MLPClassifier(hidden_layer_sizes=(10000,500), random_state=i, max_iter=200,  alpha = 0.001, early_stopping=True).fit(X_train, y_train)
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    print(i , 'train_acc:', round(train_acc,4), 'test_acc:', round(test_acc, 4))
    
    Result.append(test_acc)
    train_result.append(train_acc)
    
print('average train acc:', round(sum(train_result)/len(train_result),3))
print('average test acc:', round(sum(Result)/len(Result),3))
print('best acc:', 'test acc', round(max(Result), 3), 'train acc', round(max(train_result), 3))


# In[80]:


from sklearn.neural_network import MLPClassifier

#key_unigram

train_result = []
for i in range(0,10):
    X_train, X_test, y_train, y_test = train_test_split(data_3_t, df_1018.label, test_size=0.2, random_state=i)
    clf = MLPClassifier(hidden_layer_sizes=(10000,500), random_state=i, max_iter=200,  alpha = 0.001, early_stopping=True, learning_rate_init=0.001).fit(X_train, y_train)
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    print(i , 'train_acc:', round(train_acc,4), 'test_acc:', round(test_acc, 4))
    
    Result.append(test_acc)
    train_result.append(train_acc)
    
print('average train acc:', round(sum(train_result)/len(train_result),3))
print('average test acc:', round(sum(Result)/len(Result),3))
print('best acc:', 'test acc', round(max(Result), 3), 'train acc', round(max(train_result), 3))


# In[81]:


#key_bigram

Result = []
train_result = []
for i in range(0,10):
    X_train, X_test, y_train, y_test = train_test_split(data_4_t, df_1018.label, test_size=0.2, random_state=i)
    clf = MLPClassifier(hidden_layer_sizes=(10000,500), random_state=i, max_iter=200,  alpha = 0.001, early_stopping=True).fit(X_train, y_train)
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    print(i , 'train_acc:', round(train_acc,4), 'test_acc:', round(test_acc, 4))
    
    Result.append(test_acc)
    train_result.append(train_acc)
    
print('average train acc:', round(sum(train_result)/len(train_result),3))
print('average test acc:', round(sum(Result)/len(Result),3))
print('best acc:', 'test acc', round(max(Result), 3), 'train acc', round(max(train_result), 3))


# In[82]:


#key_unigram+bigram

Result = []
train_result = []
for i in range(0,10):
    X_train, X_test, y_train, y_test = train_test_split(data_5_t, df_1018.label, test_size=0.2, random_state=i)
    clf = MLPClassifier(hidden_layer_sizes=(10000,500), random_state=i, max_iter=200,  alpha = 0.001, early_stopping=True).fit(X_train, y_train)
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    print(i , 'train_acc:', round(train_acc,4), 'test_acc:', round(test_acc, 4))
    
    Result.append(test_acc)
    train_result.append(train_acc)
    
print('average train acc:', round(sum(train_result)/len(train_result),3))
print('average test acc:', round(sum(Result)/len(Result),3))
print('best acc:', 'test acc', round(max(Result), 3), 'train acc', round(max(train_result), 3))

