# **classification-model-for-recommendation**
# The classification model for this recommendation uses the K-Nearest Neighbors (KNN) model, this model will provide food recipe recommendations depending on user searches such as from food ingredients or the name of the food.

# **Code:**
# import library and dataset:
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline

df = pd.read_csv(r'D:\Downloads\recido.csv')
df

# Extract the required columns
titles = df['Title'].tolist()
ingredients = df['Ingredients'].tolist()
steps = df['Steps'].tolist()

# Combine title and ingredients as one text
combined_text = [f"{title} {ingredients}" for title, ingredients in zip(titles, ingredients)]

# Create KNN model
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(combined_text)
model = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='cosine')
model.fit(tfidf_matrix)

# User input
user_input_title = "Ayam"
user_input_ingredients = "Ayam"

# Transform user input into the same format as the dataset
user_input = vectorizer.transform([f"{user_input_title} {user_input_ingredients}"])

# Predict the nearest recipe based on user input
distances, indices = model.kneighbors(user_input)

# Show recipe recommendations
print("Resep Makanan Rekomendasi:")
for index in indices[0]:
    print(f"Title: {titles[index]}\nIngredients: {ingredients[index]}\nSteps: {steps[index]}\n")
