import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import jieba
import json
import re

#load tsv dataset
df_train = pd.read_csv('KDC_train.tsv', sep='\t')
print('Number of missing values:', df_train.isnull().sum())  #no missing values

#check for imbalancend classes
print('Number of OK statements:', len(df_train[df_train['status'] == 'OK'].index))  #836
print('Number of B statements:', len(df_train[df_train['status'] == 'B'].index))  #853

#undersampling to get equal numbers
rus = RandomUnderSampler(random_state=1)
df_train_bal, df_train_bal['status'] = rus.fit_resample(df_train, df_train['status'])

print('Number of balanced OK statements:', len(df_train_bal[df_train_bal['status'] == 'OK'].index))  #836
print('Number of balanced B statements:', len(df_train_bal[df_train_bal['status'] == 'B'].index))  #836

#split data into train and test, ratio 80:20
train, test = train_test_split(df_train_bal, train_size=0.70, random_state=1, shuffle=True)
train_x, train_y = train['text'], train['status']
test_x, test_y = test['text'], test['status']

#import json file (chinese stop words)
file = open('stopwords-zh.json')
stop_words_chinese = json.load(file)

#remove all non-chinese words and tokenize
def tokenize(text):
    filtrate = re.compile(u'[^\u4E00-\u9FA5]')  # non-Chinese unicode range
    text = filtrate.sub(r'', text)  # remove all non-Chinese characters
    return jieba.lcut(text)


#tokenize and vectorize data
vectorizer = TfidfVectorizer(stop_words=stop_words_chinese, tokenizer=tokenize)
X = vectorizer.fit_transform(train_x)
train_x_vector = pd.DataFrame(X.toarray(), index=train_x.index, columns=vectorizer.get_feature_names())


def tfidf_vectorization(input):
    sparse_matrix = vectorizer.transform(input)
    return pd.DataFrame(sparse_matrix.toarray(), index=input.index, columns=vectorizer.get_feature_names())

test_x_vector = tfidf_vectorization(test_x)

#check for correct dimensions
print(train_x_vector.shape)
print(test_x_vector.shape)
print(len(train_y), len(test_y))

#create model (SVM)
svc = SVC(kernel='linear')
svc.fit(train_x_vector, train_y)

print('SVM score:', svc.score(test_x_vector, test_y))

#alternative LR model
log_reg = LogisticRegression()
log_reg.fit(train_x_vector, train_y)

print('Logistic Regression score:', log_reg.score(test_x_vector, test_y))
print('Classes:', log_reg.classes_)


#load validation data
df_val = pd.read_csv('KDC_test.tsv')
val_x = df_val['text']
val_vector = tfidf_vectorization(val_x)
predicted_classes = log_reg.predict(val_vector)

df_val['status'] = predicted_classes
df_val.to_csv('KDC_test_complete.tsv', sep='\t')

