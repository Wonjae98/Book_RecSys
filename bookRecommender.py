import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt

np.seterr(invalid='ignore')

def mergeDuplicateData():
    info_original = pd.read_csv('BX-Books.csv',sep=';', on_bad_lines='skip', encoding='latin-1', low_memory=False).sort_values('ISBN')
    info_original.drop(['Book-Author','Year-Of-Publication','Publisher','Image-URL-S','Image-URL-M','Image-URL-L'],axis=1 ,inplace=True)

    ratings = pd.read_csv('BX-Book-Ratings.csv',sep=';', encoding='latin-1')
    users = pd.read_csv('BX-Users.csv', sep=';', encoding='latin-1')

    ratings = ratings[ratings['ISBN'].isin(info_original['ISBN'])]
    ratings = ratings[ratings['User-ID'].isin(users['User-ID'])].reset_index(drop=True)

    info_name_modified = info_original.copy()
    info_name_modified['Book-Title'] = info_name_modified['Book-Title'].str.replace(pat=r'[^\w]',repl=r'',regex=True)
    info_name_modified['Book-Title'] = info_name_modified['Book-Title'].str.upper()

    info_no_duplicate = info_name_modified.drop_duplicates(['Book-Title'],keep='first')
    
    for index, ISBN in enumerate(ratings['ISBN']):
        if(ISBN not in list(info_no_duplicate['ISBN'])):
            index_to_change = info_name_modified[info_name_modified['ISBN'] == ISBN].reset_index(drop=True)
            book_title = index_to_change.at[0,'Book-Title']
            new_ISBN = info_name_modified[info_name_modified['Book-Title'] == book_title].reset_index(drop=True)
            new_ISBN = new_ISBN.at[0,'ISBN']
            ratings.loc[ratings['ISBN'] == ISBN,'ISBN'] = new_ISBN
            print(index, book_title,ISBN, 'to',new_ISBN)

    for index, ISBN in enumerate(info_no_duplicate['ISBN']):
        index_to_change = info_original[info_original['ISBN'] == ISBN].reset_index(drop=True)
        book_title = index_to_change.at[0,'Book-Title']
        info_no_duplicate.loc[info_no_duplicate['ISBN'] == ISBN,'Book-Title'] = book_title
        print(index, ISBN, book_title)
        
    info_no_duplicate.to_csv("modified_book.csv",index=False)
    ratings.to_csv("modified_rating.csv",index=False)

def makeNewUserData(user_history, rating_data):
    user_data = pd.DataFrame(user_history,columns=['User-ID','ISBN','Book-Rating'])

    return pd.concat([user_data,rating_data])

def contentBasedRecommend(df, books_group, ISBN_to_recommend , num_of_recommend):
    meta = pd.read_csv('Preprocessed_data.csv',on_bad_lines='skip', low_memory=False)
    meta.drop(['Unnamed: 0','location','user_id','rating','book_title','year_of_publication','publisher','img_s','img_m','img_l','Language','city','state','country','age'],axis=1 ,inplace=True)
    meta.rename(columns = {'isbn':'ISBN'},inplace=True)
    meta = meta.sort_values('ISBN').drop_duplicates(['ISBN'],keep='first')

    df = pd.merge(df, meta, on = 'ISBN')
    df = df.loc[(df['Summary'] != '9') & (df['Category'] != '9'),:]
    df = df.loc[df['ISBN'].isin(books_group.index)].reset_index(drop=True)

    df['index'] = df.index.values
    book = df[df['ISBN'] == ISBN_to_recommend]

    to_join = ['Book-Title','book_author','Summary','Category']
    df['features'] = [' '.join(df[to_join].iloc[i,].values) for i in range(df[to_join].shape[0])]

    tf = TfidfVectorizer(stop_words='english',min_df=1,ngram_range=(1,5))
    matrix = tf.fit_transform(df['features'])
    cos= cosine_similarity(matrix)

    recommendation_list = list(enumerate(cos[book.index.values[0]]))
    recommendation_list = sorted(recommendation_list,key=lambda x:x[1],reverse=True)[1:num_of_recommend+1]

    books =[]
    for i in range(len(recommendation_list)):
        books.append([df[df['index'] == recommendation_list[i][0]]['Book-Title'].item(),recommendation_list[i][1]])

    print(pd.DataFrame(books).set_index(0))


def itemBasedRecommend(pivot_table, df,ISBN_to_recommend , num_of_components, num_of_recommend):
    scaler = MinMaxScaler()
    item_sim = cosine_similarity(pivot_table.T, pivot_table.T)

    SVD = TruncatedSVD(n_components = num_of_components, algorithm = 'arpack')
    matrix = SVD.fit_transform(pivot_table.T)

    item_sim_df = pd.DataFrame(data=item_sim,index=pivot_table.columns,columns=pivot_table.columns).drop([ISBN_to_recommend],axis = 0)
    corr= pd.DataFrame(np.corrcoef(matrix), columns = pivot_table.columns,index = pivot_table.columns).drop([ISBN_to_recommend],axis = 0)

    pivot_table = pivot_table.drop([ISBN_to_recommend],axis = 1)

    cos = pd.DataFrame(scaler.fit_transform(item_sim_df[ISBN_to_recommend][:, np.newaxis]),index=pivot_table.columns,columns=[ISBN_to_recommend])
    corr = pd.DataFrame(scaler.fit_transform(corr[ISBN_to_recommend][:, np.newaxis]),index=pivot_table.columns,columns=[ISBN_to_recommend])
    
    recommendation_list = cos*0.75 + corr*0.25
    recommendation_list = recommendation_list.sort_values(by=ISBN_to_recommend, ascending=False)[:num_of_recommend]
    
    book_title_list = []
    for i in recommendation_list.index :
        index_to_change = df[df['ISBN'] == i].reset_index(drop=True)
        book_title_list.append(index_to_change.at[0,'Book-Title'])

    recommendation_list.index = book_title_list
    
    print(recommendation_list)

def userBasedRecommend(pivot_table, df, num_of_components, num_of_recommend, want_print = True):
    ratings = pivot_table.values
    ratings_mean = np.mean(ratings,axis = 1)
    ratings = ratings - ratings_mean.reshape(-1,1)

    SVD = TruncatedSVD(n_components = num_of_components, algorithm = 'arpack')
    U = SVD.fit_transform(ratings)
    sigma=SVD.explained_variance_ratio_
    Vt= SVD.components_

    matrix_to_predict = np.dot(np.dot(U, np.diag(sigma)), Vt) + ratings_mean.reshape(-1,1)
    df_to_preditct = pd.DataFrame(matrix_to_predict, columns = pivot_table.columns,index = pivot_table.index)

    user_data = df[df['User-ID'] == 0]

    user_prediction_list = df_to_preditct.iloc[0].sort_values(ascending=False)
    recommendation_list = user_prediction_list[~user_prediction_list.index.isin(user_data['ISBN'])][:num_of_recommend]

    recomendation_ISBN = recommendation_list.copy()

    if(want_print):
        book_title_list = []
        for i in recommendation_list.index :
            index_to_change = df[df['ISBN'] == i].reset_index(drop=True)
            book_title_list.append(index_to_change.at[0,'Book-Title'])

        recommendation_list.index = book_title_list
        print('\n',recommendation_list)
    
    return recomendation_ISBN

def verifyRecommendation(ratings, info, num_of_sample, num_of_components, num_of_recommend):
    
    ratings_group = ratings.groupby('User-ID').count()
    ratings_group = ratings_group[ratings_group['ISBN'] >= 100]
    total_correct=0

    for i in range(num_of_sample):
        user_test_id = ratings_group.sample(n=1).index.values[0]

        user_test_df=ratings[ratings['User-ID']==user_test_id]
        ratings_exclude_test=ratings[ratings['User-ID']!=user_test_id]

        user_test_df=user_test_df[user_test_df['Book-Rating']>=7]
        user_test_df=user_test_df.sort_values(by='Book-Rating', ascending=False)

        user_test=user_test_df.sample(frac=0.4)
        user_test['User-ID'] = np.zeros((len(user_test),1))

        user_history=user_test.values.tolist()
    
        ratings_include_history = makeNewUserData(user_history, ratings_exclude_test)

        df = pd.merge(ratings_include_history, info, on = 'ISBN').sort_values('User-ID')

        encoder = LabelEncoder()
        df['User-ID'] = encoder.fit_transform(df['User-ID'])

        pivot_table = df.pivot_table(values='Book-Rating',index='User-ID', columns='ISBN').dropna(how = 'all').fillna(0)
    
        recommendation_list = userBasedRecommend(pivot_table, df, num_of_components, num_of_recommend, False)

        print('\n','sample', i+1) 
    
        correct = []   
        for i in recommendation_list.index :
            if(i in list(user_test_df['ISBN'])):
                correct.append([i,user_test_df[user_test_df['ISBN']==i]['Book-Rating'].values[0]])
       
        for i in correct :
            index_to_change = df[df['ISBN'] == i[0]].reset_index(drop=True)
            i[0] = index_to_change.at[0,'Book-Title']
            print(i)


        total_correct=total_correct+len(correct)
        total_num_recommend=num_of_sample*num_of_recommend
        
        print(len(correct),"/" ,num_of_recommend, "matched")

    percent=total_correct/total_num_recommend*100
    
    print()

    print("Total Matched : ", percent ,"%","(",total_correct,"/",total_num_recommend,")")



    

#0099326809 s 9
#0451118642 s
#0345243447 c 10
#059035342X h 

#to verification
num_of_sample = 100

num_of_components = 15

num_of_recommend = 20

ISBN_to_recommend = '0451118642'

user_history = [[0,'0451118642',9],
                [0,'0345243447',10],
                ]

#mergeDuplicateData()

ratings = pd.read_csv('modified_rating.csv', encoding='latin-1')
info = pd.read_csv('modified_book.csv', encoding='latin-1')

ratings['ISBN'] = ratings['ISBN'].apply(lambda x : x.zfill(10))

ratings_group = ratings.groupby('User-ID').count()
ratings_group = ratings_group[ratings_group['ISBN'] >= 10]
ratings = ratings.loc[ratings['User-ID'].isin(ratings_group.index)]

books_group = ratings.groupby('ISBN').count()
books_group = books_group[books_group['User-ID'] >= 10]
ratings = ratings.loc[ratings['ISBN'].isin(books_group.index)]

ratings = ratings.loc[ratings['Book-Rating']>0]

ratings_include_history = makeNewUserData(user_history, ratings)

df = pd.merge(ratings_include_history, info, on = 'ISBN').sort_values('User-ID')

encoder = LabelEncoder()
df['User-ID'] = encoder.fit_transform(df['User-ID'])

pivot_table = df.pivot_table(values='Book-Rating',index='User-ID', columns='ISBN').dropna(how = 'all').fillna(0)

contentBasedRecommend(info, books_group, ISBN_to_recommend, num_of_recommend)

#itemBasedRecommend(pivot_table, df, ISBN_to_recommend, num_of_components, num_of_recommend)

#userBasedRecommend(pivot_table, df, num_of_components, num_of_recommend)

#verifyRecommendation(ratings, info, num_of_sample, num_of_components, num_of_recommend)
