import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline 
from sklearn.cluster import KMeans

df = pd.read_csv("tempp.csv", encoding='cp1252')
DF = df

df.tail()

print(max(df['Books_bought']))

for i in range (len(df['Books_bought'])):
  if df.iloc[i,3] == 3000:
    new_df = df.drop([df.index[i]])

for i in range (len(new_df['Books_bought'])):
  if new_df.iloc[i,3] == 200:
    df = new_df.drop([new_df.index[i]])

print(max(df['Books_bought']))

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
# label_encoder object knows how to understand word labels.
label_encoder = preprocessing.LabelEncoder()

df['Gender'] = label_encoder.fit_transform(df['Gender'])
# scale.fit_transform(label_encoder.fit_transform(df['Gender']).reshape(1,-1))

df['Age']= label_encoder.fit_transform(df['Age'])
df['Focus']= label_encoder.fit_transform(df['Focus'])
df['Time']= label_encoder.fit_transform(df['Time'])
df['Frequency']= label_encoder.fit_transform(df['Frequency'])

targets = df[['Favourite_Book']]
df.drop(['Favourite_Book'], inplace=True, axis="columns")

df['Enjoyment_Level'] = scale.fit_transform(df['Enjoyment_Level'].to_numpy().reshape(-1,1))

df['Books_bought'] = scale.fit_transform(df['Books_bought'].to_numpy().reshape(-1,1))

df.head()

df1 = df

dummies_genre = df.Genre.str.get_dummies(', ') 
dummies_genre.head()

dummies_theme = df.Theme.str.get_dummies(', ') 
dummies_theme.head()

dummies_motivation = df.Motivation.str.get_dummies(', ') 
dummies_motivation.head()

dummies_choices = df.Choices.str.get_dummies(', ') 
dummies_choices.head()

dummies_place = df.Place.str.get_dummies(', ') 
dummies_place.head()

dummies_medium = df.Medium.str.get_dummies(', ') 
dummies_medium.head()

dummies_pop = df.Place_of_Purchase.str.get_dummies(', ') 
dummies_pop.head()

dummies_rs = df.Recommendation_Source.str.get_dummies(', ') 
dummies_rs.head()

final_df = pd.concat([df,dummies_genre,dummies_theme,dummies_motivation,dummies_choices,dummies_place,dummies_medium,dummies_pop,dummies_rs], axis="columns")
final_df.drop(['Genre', 'Theme', 'Motivation', 'Choices', 'Place', 'Medium', 'Place_of_Purchase', 'Recommendation_Source'], inplace=True, axis="columns")
final_df.head()

distortions = []
K = range(2,10,1)
for k in K:
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(final_df)
    distortions.append(kmeanModel.inertia_)

plt.figure(figsize=(16,8))
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()

from sklearn.cluster import KMeans

# for i in range(1,200,10):
#   km = KMeans(n_clusters=4,random_state=i)
#   km.fit(final_df)
#   print(i)
#   print(km.labels_)

km = KMeans(n_clusters=4,random_state=125) 
km.fit(final_df)
print(km.labels_)

for i in range(4):
  print(list(km.labels_).count(i))

y = km.labels_
final_df['Cluster'] = y.tolist()

df1, df2, df3, df4 = final_df.loc[final_df['Cluster'] == 0], final_df.loc[final_df['Cluster'] == 1], final_df.loc[final_df['Cluster'] == 2], final_df.loc[final_df['Cluster'] == 3]

for i in DF.columns:
  DF[i] = DF[i].convert_dtypes()

for i in range (len(DF['Books_bought'])):
  if DF.iloc[i,3] == 3000:
    new_DF = DF.drop([DF.index[i]])

for i in range (len(new_DF['Books_bought'])):
  if new_DF.iloc[i,3] == 200:
    DF = new_DF.drop([new_DF.index[i]])
DF['Cluster'] = y.tolist()
df1, df2, df3, df4 = DF.loc[DF['Cluster'] == 0], DF.loc[DF['Cluster'] == 1], DF.loc[DF['Cluster'] == 2], DF.loc[DF['Cluster'] == 3]

df1.drop(['Enjoyment_Level','Books_bought','Favourite_Book','Cluster'], inplace=True, axis="columns")
df1.head()

df1.dtypes
df1['Gender'] = df1['Gender'].convert_dtypes()
df1.dtypes

dummies_genre_1 = pd.DataFrame(data = final_df.loc[final_df['Cluster'] == 0].iloc[:,7:17], columns = dummies_genre.columns)
dummies_genre_2 = pd.DataFrame(data = final_df.loc[final_df['Cluster'] == 1].iloc[:,7:17], columns = dummies_genre.columns)
dummies_genre_3 = pd.DataFrame(data = final_df.loc[final_df['Cluster'] == 2].iloc[:,7:17], columns = dummies_genre.columns)
dummies_genre_4 = pd.DataFrame(data = final_df.loc[final_df['Cluster'] == 3].iloc[:,7:17], columns = dummies_genre.columns)

from mlxtend.frequent_patterns import apriori, association_rules
frq_items_1 = apriori(dummies_genre_1, min_support = 0.095, use_colnames = True)
pd.set_option('display.max_rows', None)
rules_1 = association_rules(frq_items_1, metric ="lift", min_threshold = 1)
rules_1 = rules_1.sort_values(['confidence', 'lift'], ascending =[False, False])
rules_1.head()

frq_items_2 = apriori(dummies_genre_2, min_support = 0.095, use_colnames = True)

rules_2 = association_rules(frq_items_2, metric ="lift", min_threshold = 1)
rules_2 = rules_2.sort_values(['confidence', 'lift'], ascending =[False, False])
rules_2.head()

frq_items_3 = apriori(dummies_genre_3, min_support = 0.095, use_colnames = True)

rules_3 = association_rules(frq_items_3, metric ="lift", min_threshold = 1)
rules_3 = rules_3.sort_values(['confidence', 'lift'], ascending =[False, False])
rules_3.head()

frq_items_4 = apriori(dummies_genre_4, min_support = 0.095, use_colnames = True)

rules_4 = association_rules(frq_items_4, metric ="lift", min_threshold = 1)
rules_4 = rules_4.sort_values(['confidence', 'lift'], ascending =[False, False])
rules_4.head()

def inp_preprocess(row):
  for i in row.columns:
    row[i] = row[i].convert_dtypes()
  row['Gender'] = label_encoder.fit_transform(row['Gender'])
  row['Age']= label_encoder.fit_transform(row['Age'])
  row['Focus']= label_encoder.fit_transform(row['Focus'])
  row['Time']= label_encoder.fit_transform(row['Time'])
  row['Frequency']= label_encoder.fit_transform(row['Frequency'])
  row.drop(['Favourite_Book'], inplace=True, axis="columns")
  row['Enjoyment_Level'] = scale.fit_transform(row['Enjoyment_Level'].to_numpy().reshape(-1,1))
  row['Books_bought'] = scale.fit_transform(row['Books_bought'].to_numpy().reshape(-1,1))

  genre = ["Fiction","Mystery","Adventure","Biography","Fantasy","Romance","Comics","Sci-Fi","Self Help","Other"]
  genre_dummy = pd.DataFrame(pd.np.empty((0, 10)))
  genre_dummy.set_axis(genre, axis=1, inplace=True)
  l = str(row['Gender']).split(', ')
  for i in l:
    if i in genre:
      genre_dummy.at[0, i] = 1

  theme = ["Rebellion","Vengeance","Love","Survival","Coming of age","Suffering","Power and corruption","Moral ambiguity","Other"]
  theme_dummy = pd.DataFrame(pd.np.empty((0, 9)))
  theme_dummy.set_axis(theme, axis=1, inplace=True)
  l = str(row['Theme']).split(', ')
  for i in l:
    if i in theme:
      theme_dummy.at[0, i] = 1

  motivation = ["School/College Assignments","Recommendations from a friend", "Personality Development", "Relaxation", "Enjoyment"]
  motivation_dummy = pd.DataFrame(pd.np.empty((0, 5)))
  motivation_dummy.set_axis(motivation, axis=1, inplace=True)
  l = str(row['Motivation']).split(', ')
  for i in l:
    if i in motivation:
      motivation_dummy.at[0, i] = 1

  choices = ["Well-written descriptive detail", "Witty or believable dialogue", "Interesting character arcs", "Complicated or multi-step plot", "In-depth lore that you can escape into outside the confines of a book", "Exciting action scenes", "Other"]
  choices_dummy = pd.DataFrame(pd.np.empty((0, 7)))
  choices_dummy.set_axis(choices, axis=1, inplace=True)
  l = str(row['Choices']).split(', ')
  for i in l:
    if i in choices:
      choices_dummy.at[0, i] = 1

  medium = ["Print book", "E-book", "Magazines", "Newspaper", "Applications", "Websites", "Others"]
  medium_dummy = pd.DataFrame(pd.np.empty((0, 7)))
  medium_dummy.set_axis(medium, axis=1, inplace=True)
  l = str(row['Medium']).split(', ')
  for i in l:
    if i in medium:
      medium_dummy.at[0, i] = 1
  
  place = ["In a library","While commuting","At home","In a group","Other"]
  place_dummy = pd.DataFrame(pd.np.empty((0, 5)))
  place_dummy.set_axis(place, axis=1, inplace=True)
  l = str(row['Place']).split(', ')
  for i in l:
    if i in place:
      place_dummy.at[0, i] = 1
  
  pop = ["Online, places like Amazon", "Book Stores", "Book Orders/Book Fairs from school" ,"Other"]
  pop_dummy = pd.DataFrame(pd.np.empty((0, 4)))
  pop_dummy.set_axis(pop, axis=1, inplace=True)
  l = str(row['Place_of_Purchase']).split(', ')
  for i in l:
    if i in pop:
      pop_dummy.at[0, i] = 1
  
  rs = ["Online/Social Media", "Friends", "Ads on TV/Computer", "I like to browse books on my own", "Other"]
  rs_dummy = pd.DataFrame(pd.np.empty((0, 5)))
  rs_dummy.columns = rs
  l = str(row['Recommendation_Source']).split(', ')
  for i in l:
    if i in rs:
      rs_dummy.at[0, i] = 1
      
  final_df_inp = pd.concat([row,genre_dummy,theme_dummy,motivation_dummy,choices_dummy,medium_dummy,place_dummy,pop_dummy,rs_dummy], axis="columns")
  final_df_inp.drop(['Genre', 'Theme', 'Motivation', 'Choices', 'Place', 'Medium', 'Place_of_Purchase', 'Recommendation_Source'], inplace=True, axis="columns")  
  final_df_inp.fillna(0, inplace=True)
  final_df_inp = final_df_inp.astype(int)
  return final_df_inp

rows = pd.DataFrame(DF.iloc[2])
rows = rows.to_numpy().reshape(1,17)
rows = pd.DataFrame(rows)
rows.set_axis(DF.columns, axis=1, inplace=True)
df_res = inp_preprocess(rows)

rows

cluster = km.predict(df_res)

def MatchAndRecommend(df, cluster):
  
  max1, pos = 0, -1
  genres, recommend = str(df['Genre']).split(', '), []
  
  if(cluster == 1):
    for i in range(0, len(rules_1)):
      match = 0
      for j in range(0, len(genres)):
        if(genres[j] in rules_1.iloc[i,0]):
          match += 1
      if(match > max1):
        max1 = match
        pos = i
    recommend = rules_1.iloc[pos, 1]

  elif(cluster == 2):
    for i in range(0, len(rules_2)):
        match = 0
        for j in range(0, len(genres)):
          if(genres[j] in rules_2.iloc[i,0]):
            match += 1
        if(match > max1):
          max1 = match
          pos = i
    recommend = rules_2.iloc[pos, 1]

  elif(cluster == 3):
    for i in range(0, len(rules_3)):
      match = 0
      for j in range(0, len(genres)):
        if(genres[j] in rules_3.iloc[i,0]):
          match += 1
      if(match > max1):
        max1 = match
        pos = i
    recommend = rules_3.iloc[pos, 1]

  else:
    for i in range(0, len(rules_4)):
      match = 0
      for j in range(0, len(genres)):
        if(genres[j] in rules_4.iloc[i,0]):
          match += 1
      if(match > max1):
        max1 = match
        pos = i
    recommend = rules_4.iloc[pos, 1]
  return recommend

rec = MatchAndRecommend(rows, cluster)

l1 = list(rec)
l1

DF.iloc[2]['Genre']

# Python's built-in module for encoding and decoding JSON data
import json
# Python's built-in module for opening and reading URLs
from urllib.request import urlopen

print(f"\nRECOMMENDATIONS ")
for i in l1:
  # create getting started variables
  api = "https://www.googleapis.com/books/v1/volumes?q=%27%27+subject:"
  key= "&key=AIzaSyDz32YtKjhTmi0tyvnpTHVuB8g7Ao6R78o"

  # send a request and get a JSON response
  resp = urlopen(api +i+ key)
  # parse JSON into Python as a dictionary
  book_data = json.load(resp)

  # creating additional variables for easy querying
  volume_info = book_data["items"][0]["volumeInfo"]
  author = volume_info["authors"]
  prettify_author = author if len(author) > 1 else author[0]

  print(f"\nGenre: " +i)

  # display title, author, description, print type
  print(f"\nTitle: {volume_info['title']}")
  print(f"Author: {prettify_author}")
  print(f"Description: {volume_info['description']}")
  print(f"Print Type: {volume_info['printType']}")
  print("\n***\n")
