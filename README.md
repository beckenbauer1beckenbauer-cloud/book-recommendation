# Book Recommendation Engine using K-Nearest Neighbors (KNN)

This repository contains a Machine Learning project that builds a Book Recommendation Engine using the K-Nearest Neighbors (KNN) algorithm. The model analyzes user ratings from the Book-Crossings dataset to find and recommend books that are statistically and behaviorally similar to a given title.

# 📋 Project Overview
Recommender systems are at the core of modern digital platforms like Amazon, Netflix, and Spotify. This project implements an Item-Based Collaborative Filtering approach. Instead of analyzing the textual content of the books (genres, authors, descriptions), the engine looks purely at user behavior—identifying patterns based on how different users rated different books.

Dataset Specifications
The engine is powered by the Book-Crossings Dataset, a massive social-network-style compilation containing:

1.1 Million ratings (on a scale of 1 to 10).

270,000 unique books.

90,000 unique users.

# 🔬 Mathematical & Algorithmic Foundations
K-Nearest Neighbors (KNN)
The core of this engine relies on sklearn.neighbors.NearestNeighbors. KNN treats every book as a multi-dimensional vector in a geometric space where the dimensions represent unique users, and the coordinates represent the ratings given by those users.

Cosine Similarity Metric
To determine the "closeness" or similarity between two books, the model utilizes Cosine Distance rather than standard Euclidean distance.

Cosine similarity measures the cosine of the angle between two directional vectors.

If two books receive highly similar rating patterns from the same cluster of users, the angle between their vectors approaches zero, resulting in a high similarity score (and a low cosine distance).

# 🛠️ Data Engineering & Cleaning Pipelines
Raw data in large recommender systems is notoriously noisy and sparse. Most books receive very few ratings, and many users only rate a single book. To ensure statistical significance and protect the system from computational overload, a strict filtering pipeline was implemented:

User Filtering: Users who have contributed fewer than 200 total ratings are dropped from the dataset to eliminate casual or inactive profiles.

Book Filtering: Books that have accumulated fewer than 100 total ratings are dropped to eliminate obscure or rare titles.

Pivot Transformation: The filtered tabular data is reshaped into a Pivot Table (Matrix) where:

Rows (Index): Represent unique, highly-rated Book Titles.

Columns: Represent unique, highly-active User IDs.

Cells: Represent explicit ratings. All empty values (NaN) are filled with 0 to denote that a user hasn't rated that specific book.

Memory Optimization via CSR Matrix: Because the vast majority of cells in a massive recommendation matrix are zeros, the pivot table is converted into a Compressed Sparse Row (CSR) Matrix using scipy.sparse. This compresses the data layout in memory by stripping out the zeros while retaining structural coordinates, accelerating training speeds by over 90%.

# 💻 Core Implementation
The system exposes a core function named get_recommends() which handles the text-to-vector extraction, queries the KNN model, and formats the output into a precise nested structure.

def get_recommends(book=""):
    # 1. Locate the numerical position of the text title within the matrix index
    book_index = df_pivot.index.get_loc(book)
    
    # 2. Query the trained KNN model for the 6 closest neighbors 
    # (n_neighbors=6 because index 0 is always the queried book itself)
    distances, indices = model_knn.kneighbors(
        df_pivot.iloc[book_index].values.reshape(1, -1), 
        n_neighbors=6
    )
    
    # 3. Extract, unpack, and structurally arrange the recommendations
    recommended_books = []
    
    # Loop backward from the 5th neighbor down to the 1st neighbor 
    # to maintain a clean descending distance order
    for i in range(5, 0, -1):
        book_title = df_pivot.index[indices[0][i]]
        book_distance = distances[0][i]
        recommended_books.append([book_title, book_distance])
        
    return [book, recommended_books]

# 📊 Expected Output Format
When querying the engine with a specific book title, it returns a nested list containing the original book title followed by a list of 5 recommended books alongside their precise mathematical distances:

get_recommends("Where the Heart Is (Oprah's Book Club (Paperback))")

Output:

[
  "Where the Heart Is (Oprah's Book Club (Paperback))",
  [
    ["I Know This Much Is True", 0.7677075],
    ["The Surgeon", 0.7699411],
    ["The Weight of Water", 0.77085835],
    ["I'll Be Seeing You", 0.8016211]
  ]
]

(Note: Distances near 0 indicates high similarity; distances approaching 1 indicate low similarity).

# 🚀 How to Run and Test
Open the project inside Google Colaboratory or any Jupyter Notebook environment.

Run the initial setup cells to download and extract the Book-Crossings zip files into your local directory.

Run the data cleaning, matrix reshaping, and KNN fitting cells.

Execute the testing block containing test_book_recommendation() to validate the model's performance against the strict validation assertions.


    
