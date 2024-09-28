from deepface import DeepFace
import faiss
import numpy as np
import os

def extract_embedding(image_path):
    embedding = DeepFace.represent(img_path=image_path, model_name='Facenet', enforce_detection=False)
    return np.array(embedding[0]['embedding'])

# Path to the folder containing database images
database_image_folder = "dataset"
database_image_paths = [os.path.join(database_image_folder, f) for f in os.listdir(database_image_folder) if f.endswith(('.jpg', '.png'))]

# Initialize FAISS index with Euclidean distance (L2)
embedding_dim = 128  # FaceNet embedding size
index = faiss.IndexFlatL2(embedding_dim)

# Loop through database images and add their embeddings to the FAISS index
for image_path in database_image_paths:
    image_embedding = extract_embedding(image_path).astype('float32')
    if image_embedding.shape[0] != embedding_dim:
        print(f"Warning: {image_path} has an invalid embedding size: {image_embedding.shape[0]}. Expected {embedding_dim}.")
        continue  # Skip this image if the embedding size is incorrect
    
    index.add(image_embedding.reshape(1, -1))  # Add embedding to FAISS index
# Optionally, save the FAISS index to disk for future use
faiss.write_index(index, 'image_embeddings.index')

def search_top_k(query_image_path, top_k=5):
    # Extract embedding for the query image (user's photo or live image)
    query_embedding = extract_embedding(query_image_path).astype('float32').reshape(1, -1)
    
    # Search for the top K nearest neighbors
    distances, indices = index.search(query_embedding, top_k)
    
    # Calculate similarity scores (you can invert the distance if needed)
    similarity_scores = 1 / (1 + distances)  # Example similarity calculation (not necessarily accurate)
    
    return indices, similarity_scores

# Perform search on a query image
query_image = 'test.jng'
top_k = 5
similar_image_indices, similarity_scores = search_top_k(query_image, top_k)

# Display results
for i, (index, score) in enumerate(zip(similar_image_indices[0], similarity_scores[0])):
    print(f"Top {i + 1} similar image index: {index}, Similarity Score: {score:.4f}")

