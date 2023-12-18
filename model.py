from flask import Flask, render_template, request
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
from llama_index import StorageContext, load_index_from_storage

app = Flask(__name__)

os.environ['OPENAI_API_KEY'] = 'API_KEY'
storage_context = StorageContext.from_defaults(persist_dir="Directory_containing_index")
index = load_index_from_storage(storage_context)

query_engine = index.as_query_engine()
# Load the CSV file
df = pd.read_csv('your_dataset.csv')

# Preprocess the data
df['subcategory'] = df['L2 - sub-category']
df['gender'] = df['L0 - gender']
df = df[['title', 'description', 'subcategory', 'gender', 'image_link', 'product_type', 'price', 'brand', 'season','color']]
df = df.dropna()

# Use TfidfVectorizer to convert the titles in df['title'] into TF-IDF vectors
vectorizer = TfidfVectorizer()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommendations', methods=['POST'])
def recommendations():
    user_query = request.form['user_query']
    user_query = user_query + ',Recommend products for me from my context, List them directly without any explanation as products name and each product at a line'
    user_query_response = query_engine.query(user_query)
    user_query_response = str(user_query_response)

    print('User asked:', user_query)
    print('Gpt answer:', user_query_response)
    filtered_df = filter_df_by_gender(user_query, df)
    df_matrix = vectorizer.fit_transform(filtered_df['title'])

    result_list = list_products(user_query_response)

    added_titles = set()
    recommendations_data = []
    #print(filtered_df['subcategory'].unique())
    for query_title in result_list:
        query_title = tokenize_product_title(query_title)
        query_matrix = vectorizer.transform([query_title])
        # No category filter, use the general matrix
        cosine_sim = cosine_similarity(df_matrix, query_matrix)

        # Find the index of the most similar title
        most_similar_index = cosine_sim.argmax()
        most_similar_product = filtered_df.iloc[most_similar_index]
        #if product not added , add it to the list otherwise drop
        if (most_similar_product['title'] not in added_titles and cosine_sim[most_similar_index][0] > 0.2):
            recommendations_data.append({
                "query_title": query_title,
                "most_similar_product_title": most_similar_product['title'],
                "most_similar_product_gender": most_similar_product['gender'],
                "most_similar_product_image": most_similar_product['image_link'],
                "most_similar_product_category": most_similar_product['subcategory'],
                "price":most_similar_product['price'],
                "brand":most_similar_product['brand'],
                "color":most_similar_product['color'],
                "similarity_score": cosine_sim[most_similar_index][0]
            })
            added_titles.add((most_similar_product['title']))
    return render_template('recommendations.html', recommendations=recommendations_data)

def filter_df_by_gender(user_query, df):
    unique_genders = df['gender'].unique()
    gender_pattern = r'\b(?:' + '|'.join(re.escape(g) for g in unique_genders) + r')\b'
    user_gender_match = re.search(gender_pattern, user_query, flags=re.IGNORECASE)
    user_gender = user_gender_match.group(0).lower() if user_gender_match else None
    filtered_df = df[df['gender'].str.lower().str.match(user_gender)] if user_gender else df

    print(user_gender)
    print(filtered_df['gender'].unique())
    return filtered_df

def tokenize_product_title(title):
    tokens = re.findall(r'\b(?!s\b|\b\w\b)\w+\b', title.lower())
    title = ' '.join(tokens)
    return title

def list_products(user_query_response):
    result_list = re.findall(r'-\s(.+)', user_query_response)
    if not result_list:
            result_list = user_query_response.splitlines()
    for i, outfit in enumerate(result_list, start=1):
        print(f"Product {i}: {outfit}\n")
    return result_list

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int("3000"), debug=False, threaded=True)
