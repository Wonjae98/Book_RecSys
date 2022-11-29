# MachinLearning_RecSys
We make the recommendation program using Collaborative Filtering , Content based Filtering.

We use dataset in kaggle , the link is here.

https://www.kaggle.com/datasets/mohitnirgulkar/book-recommendation-data?select=BX-Books.csv



## Function Definition

#### makeNewUserData( user_history, rating_data ):
 : Make the new dataframe with user_history that who want recommendation


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
  - want_print(boolean) : if true, print the recommendation list


### Test function of our model
#### verifyRecommendation(ratings, info, num_of_sample, num_of_components, num_of_recommend):
- Divide ratings dataset up into the test user dataset and compare recommend books.

  - ratings : dataset that has column ['User-ID'], ['ISBN'] , ['ratings']
  - info : dataset that has column ['Title'], ['ISBN'] 
  - num_of_sample : The number of sample that you wanna check
  - num_of_components : desired dimensionality of output data in TruncatedSVD
  - num_of_recommend : the number of book that we will recommend
