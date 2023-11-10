import pandas as pd

pd.set_option('display.max_columns', None)
X = pd.read_csv("train.tsv.gz", sep="\t")
Xtest = pd.read_csv("test.tsv.gz", sep="\t")


X['raw_branded_description'] = X['raw_branded_description'].fillna("")
X['lemmaized_wo_stopwords_raw_branded_description'] = X['lemmaized_wo_stopwords_raw_branded_description'].fillna("")

Xtest['raw_branded_description'] = Xtest['raw_branded_description'].fillna("")
Xtest['lemmaized_wo_stopwords_raw_branded_description'] = Xtest['lemmaized_wo_stopwords_raw_branded_description'].fillna("")


X['employer_id'] = X['employer_id'].fillna(-1)
X['unified_address_region'] = X['unified_address_region'].fillna('NoneInformation')
X['unified_address_city'] = X['unified_address_city'].fillna('NoneInformation')

Xtest['employer_id'] = Xtest['employer_id'].fillna(-1)
Xtest['unified_address_region'] = Xtest['unified_address_region'].fillna('NoneInformation')
Xtest['unified_address_city'] = Xtest['unified_address_city'].fillna('NoneInformation')


X['accept_handicapped'] = X['accept_handicapped'].astype(int)
X['accept_kids'] = X['accept_kids'].astype(int)

Xtest['accept_handicapped'] = Xtest['accept_handicapped'].astype(int)
Xtest['accept_kids'] = Xtest['accept_kids'].astype(int)


X['employer_id'] = X['employer_id'].astype(int)
Xtest['employer_id'] = Xtest['employer_id'].astype(int)

X["key_skills_name"] = X["key_skills_name"].apply(lambda x: "" if x == "не указано" else x)
Xtest["key_skills_name"] = Xtest["key_skills_name"].apply(lambda x: "" if x == "не указано" else x)

X["if_foreign_language"] = X["if_foreign_language"].apply(lambda x: 1 if x == "Указано" else 0)
X["is_branded_description"] = X["is_branded_description"].apply(lambda x: 1 if x == "заполнено" else 0)

Xtest["if_foreign_language"] = Xtest["if_foreign_language"].apply(lambda x: 1 if x == "Указано" else 0)
Xtest["is_branded_description"] = Xtest["is_branded_description"].apply(lambda x: 1 if x == "заполнено" else 0)


to_drop = ['name',
           'unified_address_country',
           'raw_description',
           'raw_branded_description']
X = X.drop(to_drop, axis=1)
Xtest = Xtest.drop(to_drop, axis=1)

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
for i in ['experience_name', 'schedule_name', 'unified_address_city', 'unified_address_state', 'unified_address_region',
          'specializations_profarea_name', 'employment_name', 'employer_industries']:
    X[i] = le.fit_transform(X[i])
    Xtest[i] = le.fit_transform(Xtest[i])

import re
from nltk.corpus import stopwords
from nltk import word_tokenize
import pymorphy2
import nltk
nltk.download('stopwords')
russian_stopwords = stopwords.words("russian")
morph = pymorphy2.MorphAnalyzer()

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(min_df=5, max_df=1000)

train_text = []
test_text = []
for i in ['employer_name', 'key_skills_name', 'professional_roles_name', 'languages_name',
          'lemmaized_wo_stopwords_raw_description', 'lemmaized_wo_stopwords_raw_branded_description', 'name_clean']:
    X[i] = X[i].apply(lambda x: x.lower())
    X[i] = X[i].apply(lambda x: re.sub(r'[^A-Za-zА-Яа-яЁё -]', '', x))

    Xtest[i] = Xtest[i].apply(lambda x: x.lower())
    Xtest[i] = Xtest[i].apply(lambda x: re.sub(r'[^A-Za-zА-Яа-яЁё -]', '', x))

    if i in ['employer_name', 'key_skills_name', 'professional_roles_name', 'languages_name']:
        X[i] = X[i].apply(lambda x: " ".join([j for j in word_tokenize(x) if j not in russian_stopwords]))
        X[i] = X[i].apply(lambda x: " ".join([morph.parse(j)[0].normal_form for j in word_tokenize(x)]))

        Xtest[i] = Xtest[i].apply(lambda x: " ".join([j for j in word_tokenize(x) if j not in russian_stopwords]))
        Xtest[i] = Xtest[i].apply(lambda x: " ".join([morph.parse(j)[0].normal_form for j in word_tokenize(x)]))
    train_text.append(vectorizer.fit_transform(X[i]))
    test_text.append(vectorizer.transform(Xtest[i]))

stacked_da = X[['experience_name', 'schedule_name', 'accept_handicapped', 'accept_kids', 'unified_address_city',
                'unified_address_state', 'unified_address_region', 'specializations_profarea_name', 'if_foreign_language',
                'is_branded_description', 'employment_name', 'employer_id', 'employer_industries']]
stacked_da_test = Xtest[['experience_name', 'schedule_name', 'accept_handicapped', 'accept_kids', 'unified_address_city',
                'unified_address_state', 'unified_address_region', 'specializations_profarea_name', 'if_foreign_language',
                'is_branded_description', 'employment_name', 'employer_id', 'employer_industries']]

from scipy.sparse import vstack
stacked_data = vstack([train_text[0], train_text[1], train_text[2], train_text[3], train_text[4], train_text[5], train_text[6], stacked_da])
stacked_data_test = vstack([test_text[0], test_text[1], test_text[2], test_text[3], test_text[4], test_text[5], test_text[6], stacked_da_test])


from sklearn.linear_model import Ridge
clf = Ridge(alpha=1.0, random_state=241)
clf.fit(stacked_data, X['salary_mean_net'])
predict = clf.predict(stacked_data_test)
pd.DataFrame({'id': Xtest['id'], 'salary_mean_net': predict}).to_csv('aaa.tsv', sep="\t", index=False)
