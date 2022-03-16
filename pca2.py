import matplotlib.pyplot as plt
import numpy as np

import pandas as pd

data=pd.io.parsers.read_csv('ratings.dat',names=['user_id','movie_id','rating','time'],engine='python',delimiter='::')
movie_data=pd.io.parsers.read_csv('movies.dat',names=['movie_id','title','genre'],engine='python',delimiter='::')

ratings_mat=np.zeros(shape=(np.max(data.movie_id.values),np.max(data.user_id.values)))
ratings_mat[data.movie_id.values-1,data.user_id.values-1]=data.rating.values

normalized_mat= ratings_mat - np.asarray([(np.mean(ratings_mat,1))]).T

A=normalized_mat / np.sqrt(ratings_mat.shape[0]-1)
U,S,V=np.linalg.svd(A)

def top_cosine_similarity(data,movie_id,top_n):
    index=movie_id-1
    movie_row=data[index,:]
    magnitude=np.sqrt(np.einsum('ij,ij->i',data,data))
    similarity=np.dot(movie_row,data.T) / (magnitude[index]*magnitude)
    sort_index=np.argsort(-similarity)
    return sort_index[:top_n]

def print_similar_movies(movie_data,movie_id,top_index):
    print('recommended movies for {0}: \n'.format(movie_data[movie_data.movie_id==movie_id].title.values[0]))
    for id in top_index+1:
        print(movie_data[movie_data.movie_id==id].title.values[0])
k=10
movie_id=3952
top_n=10
sliced=U[:,:k]
indexes=top_cosine_similarity(sliced,movie_id,top_n)
print_similar_movies(movie_data,movie_id,indexes)


