# Netflix Movie Recommendation System (SVD)

A machine learning project that implements a personalized recommendation engine using Matrix Factorization techniques. By applying Singular Value Decomposition (SVD), the system uncovers latent features in user-item interactions to predict how a user would rate a movie they haven't seen yet.

## Overview

Recommendation engines are the backbone of modern streaming services. This project moves beyond simple "popularity-based" suggestions to "Collaborative Filtering." By analyzing a large matrix of user ratings, the SVD algorithm identifies hidden patterns (latent factors) such as genre preferences, actor affinity, and directorial style to provide highly accurate, individualized recommendations.

## Dataset

- **Source:** MovieLens/Netflix inspired dataset (`27_movies.csv` and accompanying ratings).
- **Key Files:**
  - `movies.csv`: Contains `movieId`, `title`, and `genres`.
  - `ratings.csv`: User-contributed ratings for various movies.
- **Scale:** Analysis of thousands of movies and user interactions to create a sparse user-item matrix.

## Objectives

- Implement **Collaborative Filtering** to handle the "cold start" and personalization challenges.
- Use **Singular Value Decomposition (SVD)** to factorize the user-movie matrix.
- Clean and merge datasets to map movie IDs to titles and genres.
- Predict estimated ratings for specific users and output a "Top 10" recommendation list.
- Evaluate model performance using **Root Mean Squared Error (RMSE)**.

## Methods and Analysis

The project follows a structured recommendation pipeline:

- **Data Wrangling**
  - Merging movie metadata with user ratings.
  - Filtering for active users and frequently rated movies to reduce noise and computational load.

- **The SVD Algorithm**
  - Decomposing the user-item matrix into three matrices: $U$ (user-latent features), $\Sigma$ (singular values), and $V^T$ (item-latent features).
  - Using the **Surprise Library** to handle the matrix factorization process.



- **Prediction & Scoring**
  - Calculating the `Estimated_Score` for movies a user has not yet watched.
  - Sorting results to present the highest-rated unseen content.



- **Model Validation**
  - Using Cross-Validation to ensure the SVD model generalizes well across different subsets of the data.
  - Metrics: Achieving a low **RMSE**, indicating that predicted ratings are close to actual user preferences.

## Tech Stack

- **Language:** Python 3
- **Libraries:**
  - `pandas` and `numpy`: Data manipulation.
  - `surprise`: Specialized library for scikit-learn compatible recommender systems (SVD).
  - `matplotlib` and `seaborn`: Visualization of rating distributions.
- **Environment:** Jupyter / Google Colab

## How to Run

1. **Clone this repository:**
   ```bash
   git clone [https://github.com/](https://github.com/)<your-username>/netflix-recommendation-svd.git
   cd netflix-recommendation-svd

2. *Create and activate a virtual environment (optional but recommended):*
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate

3. *Install dependencies:*
   pip install pandas numpy scikit-surprise matplotlib seaborn

4.  Ensure the dataset is present: Place 27_movies.csv and your ratings file in the root folder.

5. *Open the notebook:*
   jupyter notebook 27_Netflix_recommendation_Project_SVD.ipynb
