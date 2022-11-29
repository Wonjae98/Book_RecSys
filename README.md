# MachinLearning_RecSys

## Function Definition

makeNewUserData( user_history, rating_data ):
 : Make the new dataframe with user_history that who want recommendation


### Collaborative Filtering ( Item-based)


#### bookRecommend(pivot_table,df,ISBN_to_recommend, num_of_components):
- Recommend the book similar with the input book

  - pivot_table : pivot table ( index = 'USER-ID' / values = 'Book-Rating' / columns ='ISBN')
  - df : dataFrame
  - ISBN_to_recommend(String) : The value of ISBN (For Finding similar books with it)
  - num_of_components(int) : desired dimensionality of output data in TruncatedSVD


### Collaborative Filtering ( User-based)
#### userRecommend(pivot_table, df, num_of_components, num_of_recommend, want_print):
- Recommend the book the input user will like

  - pivot_table : pivot table ( index = 'USER-ID' / values = 'Book-Rating' / columns ='ISBN')
  - df : dataFrame
  - num_of_components(int) : desired dimensionality of output data in TruncatedSVD
  - num_of_recommend(int) : the number of books that will recommend
  - want_print(boolean) : if true, print the recommendation list

#### verifyRecommendation(ratings, info, num_of_sample, num_of_components, num_of_recommend):
- Make 
