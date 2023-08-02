import faiss
import streamlit as st
import pandas as pd
import time
from tqdm import tqdm
import seaborn as sns
import numpy as np
from textblob import TextBlob
from PIL import Image
from bert_score import score
from sentence_transformers import SentenceTransformer
logo_image_path = "C:\\Users\\manan atul doshi\\Desktop\\Project ON NLP\\Logo for Compunnel.png"
logo_image = Image.open(logo_image_path)
resized_logo = logo_image.resize((100, 100))
st.set_page_config(page_title = "Compunnel digital",page_icon=resized_logo,layout="wide")

model = SentenceTransformer('msmarco-distilbert-base-dot-prod-v3')

def fetch_movie_info(dataframe_idx, movie):
    info = movie.iloc[dataframe_idx]
    meta_dict = {}
    meta_dict['Title'] = info['Title']
    meta_dict['Plot'] = info['Plot'][:500]
    return meta_dict

def search(query, top_k, index, movie, model):
    t=time.time()
    query_vector = model.encode([query])
    top_k = index.search(query_vector, top_k)
    print('>>>> Results in Total Time: {}'.format(time.time()-t))
    top_k_ids = top_k[1].tolist()[0]
    top_k_ids = list(np.unique(top_k_ids))
    results =  [fetch_movie_info(idx, movie) for idx in top_k_ids]
    return results

def main():
    st.image(resized_logo, use_column_width=False)
    st.title("Document Releivance Search")

    # Upload the CSV file
    csv_file = st.sidebar.file_uploader("Upload the CSV file", type="csv")

    if csv_file is not None:
        # Read the CSV file
        movie = pd.read_csv(csv_file)
        movie = movie[['Title','Plot']]

        # Load the index file
        index = faiss.read_index('C:\\Users\\manan atul doshi\\Desktop\\Project ON NLP\\movie_plot.index')

        query = st.text_input("Enter a query :")
        top_k = st.number_input("Enter the number of results:", 1, 10, 5)

        button = st.button("Search")

        if query and top_k and button:
            results = search(query, top_k, index, movie, model)

            # Output the results on the main page
            st.write("Results (Faiss):")
            st.table(results)

            ranked_results_bert = []
            ref=[query]
            for cand in results:
                P, R, F1 = score([cand['Plot'][:1000]], ref, lang='en')
                ranked_results_bert.append({'Title': cand['Title'], 'Score': F1.numpy()[0]})

            ranked_results_bert = sorted(ranked_results_bert, key=lambda x: x['Score'], reverse=True)

            # Output the results on the main page
            st.write("Results (BERT Score):")
            st.table(ranked_results_bert)

if __name__ == "__main__":
    main()
