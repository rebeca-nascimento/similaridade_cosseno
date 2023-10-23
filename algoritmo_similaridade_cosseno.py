import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords")

df = pd.read_csv("C:/Users/thami/OneDrive/Documentos/Algoritmo/movies_final.csv")
movie_name = df["title"].tolist()
synopsis = df[("overview")].tolist()
stop_words = set(stopwords.words("english"))
vectorizer = CountVectorizer() 
vectors_count = vectorizer.fit_transform(synopsis) 
user_input = input("Digite a descrição do filme que deseja comparar:\n")
input_vector = vectorizer.transform([user_input])
similarity = cosine_similarity(input_vector, vectors_count) 
top_similares = similarity[0].argsort()[::-1][:5]

print("\nAs cinco descrições mais similares são: ")
for i, indice in enumerate(top_similares):
    print(f"\nFilme: {movie_name[indice]}")
    print(f"Descrição: {synopsis[indice]}")
    angulo_cos = np.arccos(similarity[0][indice])
    angulo_graus = np.degrees(angulo_cos)
    print(f"Ângulo do Cosseno (em graus): {angulo_graus} graus\n")