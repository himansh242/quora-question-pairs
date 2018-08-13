import pandas as pd
df = pd.read_csv('../Desktop/Quora/df2.csv',encoding="ISO-8859-1 ")
df2 = df.copy()
#df2

del df2['Unnamed: 0']
pd.set_option('display.max_colwidth', -1)
pd.set_option('display.max_rows', 50)

#!pip install pycorenlp

from pycorenlp import StanfordCoreNLP
nlp = StanfordCoreNLP('http://localhost:9000')
question1_list = df2['question1'].astype(str).tolist()

output_list = []
for question in question1_list:
    output_list.append(nlp.annotate(question, properties={
  'annotators': 'tokenize,truecase,pos,ner,regexner',
  'outputFormat': 'json'
  }))
    print(question1_list.index(question))
for x in output_list:
    if type(x) == str:
        print(output_list.index(x))



output_list[173759] = nlp.annotate(q1_list[173759], properties={
  'annotators': 'tokenize,truecase,pos,ner,regexner',
  'outputFormat': 'json'
  })



q1_list = df2['question1'].tolist()

import csv
rows = zip(q1_list, output_list)
with open("q1_annotated.csv",'w') as resultFile:
    wr = csv.writer(resultFile, dialect='excel')
    for row in rows:
        wr.writerow(row)



import itertools
q1_annotated = dict(zip(q1_list,output_list))



def get_lemma(string,annotated_dict):
    lemma = [] 
    for token in annotated_dict[string]['sentences'][0]['tokens']:
        lemma.append(token['lemma'])
    return ' '.join(lemma)

df2['q1_lemma'] = df2['question1'].astype(str).apply(lambda x: get_lemma(x, q1_annotated))

def get_postag(string,annotated_dict):
    postag = [] 
    for token in annotated_dict[string]['sentences'][0]['tokens']:
        postag.append(token['pos'])
    return postag
df2['q1_pos'] = df2['question1'].astype(str).apply(lambda x: get_postag(x, q1_annotated))

def entity_recognition(string, annotated_dict):
    word = []
    ner = [] 
    for token in annotated_dict[string]['sentences'][0]['tokens']:
        if token['ner'] != 'O':
            word.append(token['word'])
            ner.append(token['ner'])
    return [x for x in zip(ner,word)]

df2['q1_ner'] = df2['question1'].astype(str).apply(lambda x: entity_recognition(x, q1_annotated))

question2_list = df2['question2'].astype(str).tolist()

output_list2 = []
for question in question2_list:
    output_list2.append(nlp.annotate(question, properties={
      'annotators': 'tokenize,truecase,pos,ner,regexner',
      'outputFormat': 'json'
      }))
    print(question2_list.index(question))



import csv
rows = zip(question2_list, output_list2)
with open("q2_annotated.csv",'w') as resultFile:
    wr = csv.writer(resultFile, dialect='excel')
    for row in rows:
        wr.writerow(row)



import itertools
q2_annotated = dict(zip(question2_list,output_list2))



df2['q2_lemma'] = df2['question2'].astype(str).apply(lambda x: get_lemma(x, q2_annotated))
df2['q2_pos'] = df2['question2'].astype(str).apply(lambda x: get_postag(x, q2_annotated))
df2['q2_ner'] = df2['question2'].astype(str).apply(lambda x: entity_recognition(x, q2_annotated))

df2.to_csv('df2_june11.csv')



df = pd.read_csv("/Users/siliwang/Desktop/Desktop/职业/MSOR/2017Summer/Projects/Quora/df2_june11.csv",encoding="ISO-8859-1 ")
df2 = df.copy()
del df2['Unnamed: 0']
#df2

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(analyzer = "word",   
                             tokenizer = None,    
                             preprocessor = None, 
                             stop_words = None)




import ast
q1_postag_list = [' '.join(ast.literal_eval(x)) for x in df2['q1_pos']]
q1_vectors = vectorizer.fit_transform(q1_postag_list)
q1_vectorarrays = q1_vectors.toarray()
q2_postag_list = [' '.join(ast.literal_eval(x)) for x in df2['q2_pos']]
q2_vectors = vectorizer.fit_transform(q2_postag_list)
q2_vectorarrays = q2_vectors.toarray()

def get_cosine_distance(array1, array2):
    import numpy as np
    from numpy import linalg as LA
    return np.dot(array1,np.transpose(array2))/(LA.norm(array1)*LA.norm(array2))


get_cosine_distance(q1_vectorarrays[0], q2_vectorarrays[0])




def get_pos_cos_dist_list(array1, array2):
    cos_dist_list = []
    for i in range(len(array1)):
        cos_dist_list.append(get_cosine_distance(array1[i],array2[i]))
    return cos_dist_list

pos_cosine_list = get_pos_cos_dist_list(q1_vectorarrays,q2_vectorarrays)
df2['pos_cosine'] = pd.Series(pos_cosine_list).values




def get_euclidean_distance(array1,array2):
    import numpy as np
    from numpy import linalg as LA
    return LA.norm(array1-array2)

get_euclidean_distance(q1_vectorarrays[0],q2_vectorarrays[0])

def get_pos_euclidean_list(array1,array2):
    euclidean_list = []
    for i in range(len(array1)):
        euclidean_list.append(get_euclidean_distance(array1[i],array2[i]))
    return euclidean_list



pos_euclidean_list = get_pos_euclidean_list(q1_vectorarrays,q2_vectorarrays)
df2['pos_euclidean'] = pd.Series(pos_euclidean_list).values

q1_entity_list = df2['q1_ner'].apply(lambda x: ast.literal_eval(x)).tolist()
q2_entity_list = df2['q2_ner'].apply(lambda x: ast.literal_eval(x)).tolist()

entity_list = q1_entity_list + q2_entity_list

entity_types = set([y[0] for x in entity_list for y in x])
len(entity_types)

entity_types = list(entity_types)

country_set = set([y[1] for x in entity_list for y in x if y[0] == 'COUNTRY'])
country_set

city_set = set([y[1] for x in entity_list for y in x if y[0] == 'CITY'])
city_set




def entity_list_by_category(entity_types, entity_list):
    entity_dict = {}
    for entity_type in entity_types:
        entity_dict[entity_type] = set([y[1].lower() for x in entity_list for y in x if y[0] == entity_type])
    return entity_dict
entity_dict = entity_list_by_category(entity_types, entity_list)

entity_dict['IDEOLOGY']

def get_editdistance_vecnorm(entity_tuple_list1, entity_tuple_list2, entity_types):
    import editdistance
    import numpy as np
    from numpy import linalg as LA
    
    entity_tuple_dict1 = dict(ast.literal_eval(entity_tuple_list1))
    entity_tuple_dict2 = dict(ast.literal_eval(entity_tuple_list2))
    
    name_list1 = list(' '*len(entity_types))
    name_list2 = list(' '*len(entity_types))
    
    for entity_type in entity_types:
        index = entity_types.index(entity_type)
        try:
            name_list1[index] = entity_tuple_dict1[entity_type].lower()
        except:
            name_list1[index] = ' '
            
    for entity_type in entity_types:
        index = entity_types.index(entity_type)
        try:
            name_list2[index] = entity_tuple_dict2[entity_type].lower() 
        except:
            name_list2[index] = ' '
            
    #print(name_list1)
    #print(name_list2)
    
    edit_dist_vector = []
    for i in range(len(name_list1)):



        edit_dist_vector.append(editdistance.eval(name_list1[i],name_list2[i]))
    #print(edit_dist_vector)
    
    return LA.norm(np.asarray(edit_dist_vector))


get_editdistance_vecnorm(df2['q1_ner'][4],df2['q2_ner'][4],entity_types)


df2['entity_distance'] = df2.apply(lambda row: get_editdistance_vecnorm(row['q1_ner'], row['q2_ner'], entity_types), axis = 1)




