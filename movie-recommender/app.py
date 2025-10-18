from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import numpy as np
import difflib
import os

app = Flask(__name__)

# --- CONFIG: path to your CSV (adjust if needed) ---
DATA_PATH = "./mnt/data/netflix_data.csv"  # developer message said it's uploaded here

# --- Load dataset and build TF-IDF model at startup ---
print("Loading dataset...", flush=True)
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Please check path.")

netflix_data = pd.read_csv(DATA_PATH)

# Ensure 'title' and 'description' columns exist
if 'title' not in netflix_data.columns:
    raise KeyError("'title' column not found in the CSV.")
if 'description' not in netflix_data.columns:
    # If description missing, create empty column
    netflix_data['description'] = ''

# --- Combine multiple text features for better recommendations ---
print("Preparing combined text features...", flush=True)

# Fill missing values for all text columns we want to use
for col in ['title', 'cast', 'listed_in', 'description']:
    if col not in netflix_data.columns:
        netflix_data[col] = ''
    netflix_data[col] = netflix_data[col].fillna('')

# Create a new column combining important textual features
netflix_data['combined_features'] = (
    netflix_data['title'] + ' ' +
    netflix_data['cast'] + ' ' +
    netflix_data['listed_in'] + ' ' +
    netflix_data['description']
)

print("Computing TF-IDF matrix (this may take a little while)...", flush=True)
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf_matrix = vectorizer.fit_transform(netflix_data['combined_features'])

print("Computing cosine similarity matrix...", flush=True)
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

print("Computing cosine similarity matrix...", flush=True)
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Map movie titles to indices (safe - drop duplicates keeping first)
indexes = pd.Series(netflix_data.index, index=netflix_data['title']).drop_duplicates()

# --- Helper functions ---

def get_closest_title(query_title, n_matches=1, cutoff=0.5):
    """
    Use difflib to find close matches. Returns best match or None.
    """
    titles = netflix_data['title'].astype(str).tolist()
    matches = difflib.get_close_matches(query_title, titles, n=n_matches, cutoff=cutoff)
    return matches[0] if matches else None

def get_recommendations(movie_title, similarity_matrix=cosine_sim, top_n=10):
    """
    Given a movie title, return top_n similar movie titles.
    Raises KeyError if exact title not found.
    """
    if movie_title not in indexes:
        raise KeyError(f"Title '{movie_title}' not found in index.")
    movie_idx = indexes[movie_title]
    # enumerate similarity scores for that movie row
    similarity_scores = list(enumerate(similarity_matrix[movie_idx]))
    # sort by score descending
    similarity_scores = sorted(similarity_scores, key=lambda pair: pair[1], reverse=True)
    # skip the first one (itself) and pick next top_n
    top_scores = similarity_scores[1: top_n + 1]
    recommended_indices = [pair[0] for pair in top_scores]
    return netflix_data['title'].iloc[recommended_indices].tolist()

# --- Routes ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    """
    Expects JSON: { "query": "movie name" }
    Returns JSON: { "query_used": "...", "results": [...], "message": "..." }
    """
    data = request.get_json(force=True)
    query = data.get('query', '').strip()
    if not query:
        return jsonify({"results": [], "message": "Please provide a movie name."}), 400

    # try exact match first
    try:
        results = get_recommendations(query, top_n=10)
        return jsonify({"query_used": query, "results": results, "message": "Exact match found"})
    except KeyError:
        # try fuzzy match
        close = get_closest_title(query, cutoff=0.45)
        if close:
            try:
                results = get_recommendations(close, top_n=10)
                return jsonify({"query_used": close, "results": results, "message": f"No exact match. Showing results for '{close}' (closest match)."})
            except Exception as e:
                return jsonify({"results": [], "message": f"Error when recommending for matched title: {str(e)}"}), 500
        else:
            # no close match
            return jsonify({"results": [], "message": "Movie not found. Try a different title or check spelling."}), 404

if __name__ == '__main__':
    # For development only. In production use gunicorn or similar.
    app.run(host='0.0.0.0', port=5000, debug=True)
