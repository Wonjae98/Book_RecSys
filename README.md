# MachinLearning_RecSys
We make the book recommendation program using Content-based Filtering, Item-based Collaborative Filtering , User-based Collaborative Filtering.

We use dataset in kaggle , the link is here.

https://www.kaggle.com/datasets/mohitnirgulkar/book-recommendation-data?select=BX-Books.csv
https://www.kaggle.com/datasets/ruchi798/bookcrossing-dataset


## Architecture 

<img width="1156" alt="image" src="https://user-images.githubusercontent.com/107402065/204488629-6a7cd571-0ed7-4e87-bb83-3fb66888d40d.png">

## Function Definition


#### mergeDuplicatData():
- Make the original dataset who has duplicate data into clean dataset that we can use 

#### makeNewUserData( user_history, rating_data ):
 - Make the new dataframe with user_history that who want recommendation

### Contents-based Filtering

#### contentBasedRecommend(df, books_group, ISBN_to_recommend, num_of_recommend):
- Recommend the book similar with the book ( ISBN_ to _ recommend)
- For this, We use meta data which has many features( Book title, book_author, summary, category,)
- We use TF-IDF vectorizer for analyzing about features, and use cosine similiarity of them.
  
  
  - df : dataframe
  - book_group : only books have been evaluated mor than 10 times
  - ISBN_to_recommend : the book input for recommend 
  - num_of_recommend : the number of books that we will recommend

  
### Collaborative Filtering ( Item-based )


#### bookRecommend(pivot_table,df,ISBN_to_recommend, num_of_components):
- Recommend the book similar with the input book

  - pivot_table : pivot table ( index = 'USER-ID' / values = 'Book-Rating' / columns ='ISBN')
  - df : dataFrame
  - ISBN_to_recommend(String) : The value of ISBN (For Finding similar books with it)
  - num_of_components(int) : desired dimensionality of output data in TruncatedSVD


### Collaborative Filtering ( User-based )
#### userRecommend(pivot_table, df, num_of_components, num_of_recommend, want_print):
- Recommend the book the input user will like

  - pivot_table : pivot table ( index = 'USER-ID' / values = 'Book-Rating' / columns ='ISBN')
  - df : dataFrame
  - num_of_components(int) : desired dimensionality of output data in TruncatedSVD
  - num_of_recommend(int) : the number of books that we will recommend
  - want_print(boolean, default=True) : if true, print the recommendation list


### Test function of our model
#### verifyRecommendation(ratings, info, num_of_sample, num_of_components, num_of_recommend):
- Divide ratings dataset up into the test user dataset and compare recommend books.
- You can only use this function in user-based collaborative filtering

  - ratings : dataset that has column ['User-ID'], ['ISBN'] , ['ratings']
  - info : dataset that has column ['Title'], ['ISBN'] 
  - num_of_sample : The number of sample that you wanna check
  - num_of_components : desired dimensionality of output data in TruncatedSVD
  - num_of_recommend : the number of book that we will recommend
