# MachinLearning_RecSys

### Function Definition

1. makeNewUserData( user_history, rating_data ):
  Make the new dataframe with user_history that who want recommendation

2. bookRecommend(pivot_table,df,ISBN_to_recommend, num_of_components):
    Collaborative Filtering ( Item-based)
    
    Recommend the book similar with the input book
    pivot_table : pivot table ( index = 'USER-ID' / values = 'Book-Rating' / columns ='ISBN')
    df : dataFrame
    ISBN_to_recommend : The value of ISBN (For Finding similar books with it)
    num_of_components : desired dimensionality of output data in TruncatedSVD


3. userRecommend(pivot_table, df, num_of_components, num_of_recommend, want_print):
  Collaborative Filtering ( User-based)
  
  Recommend the book the input user will like
  pivot_table : pivot table ( index = 'USER-ID' / values = 'Book-Rating' / columns ='ISBN')
  df : dataFrame

4. verifyRecommendation(ratings, info, num_of_sample, num_of_components, num_of_recommend):
