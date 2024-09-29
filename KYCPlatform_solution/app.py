from flask import Flask, request, jsonify
from deepface import DeepFace
import numpy as np
from werkzeug.utils import secure_filename
import os
import cv2
import faiss

app = Flask(__name__)



def extract_faces(image_path):
    # Read the image
    image = cv2.imread(image_path)

    if image is None:
        print("Error: Could not read the image.")
        return None

    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Load the face cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        print("No faces found.")
        return None

    # Ensure the 'face_extracted' folder exists
    face_dir = 'face_extracted'
    if not os.path.exists(face_dir):
        os.makedirs(face_dir)

    # Save the first extracted face (if multiple faces are detected, just take the first one)
    for (x, y, w, h) in faces:
        face = image[y:y + h, x:x + w]  # Extract the face
        face_image_path = os.path.join(face_dir, 'extracted_face' + os.path.splitext(image_path)[1])
        cv2.imwrite(face_image_path, face)
        return face_image_path

    return None




 

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed extensions (just as a safety measure, but you can tweak this as needed)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Create the uploads folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Utility function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route to handle the image upload
@app.route('/upload', methods=['POST'])
def upload_image():
    # Check if the post request has the file part
    if 'image' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['image']
    
    # If no file is selected
    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400
    
    # Check if the file has an allowed extension
    if file and allowed_file(file.filename):
        # Get the file extension
        file_extension = file.filename.rsplit('.', 1)[1].lower()
        
        # Create the new filename with the original extension
        new_filename = f'live_image.{file_extension}'
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], new_filename)
        
        # Save the file
        file.save(file_path)
        
        # Return a success response with the file path
        return jsonify({
            'message': 'File successfully uploaded',
            'file_path': file_path
        }), 200
    else:
        return jsonify({'error': 'File type not allowed'}), 400


# Define a function to extract embeddings using DeepFace
def extract_embedding(image_path):
    try:
        # DeepFace expects the path of the image or numpy array, so ensure the path is passed
        embedding = DeepFace.represent(img_path=image_path, model_name='Facenet', enforce_detection=False)
        return embedding[0]['embedding']
    except Exception as e:
        print(f"Error during embedding extraction: {str(e)}")
        return None

# Define a function to compare two embeddings manually
def compare_embeddings(embedding1, embedding2, threshold=0.7):
    # Euclidean distance between two embeddings
    dist = np.linalg.norm(np.array(embedding1) - np.array(embedding2))
    return dist < threshold

@app.route('/verify_identity', methods=['POST'])
def verify_identity():
    if 'id_image' not in request.files or 'second_image' not in request.files:
        return jsonify({"status": "failure", "message": "Both ID image and second image are required."}), 400
    
    # Retrieve the images from the request
    id_imag = request.files['id_image']
    second_image = request.files['second_image']

    
    # Ensure the 'uploads' folder exists
    if not os.path.exists('uploads'):
        os.makedirs('uploads')

    # Save the uploaded images temporarily
    # filename_id = 'id_image.jpg'
    # filename_second = 'second_image.jpg'
    id_image_path = os.path.join("uploads", id_imag.filename)  # Correct attribute
    second_image_path = os.path.join("uploads", 'second_image' + os.path.splitext(second_image.filename)[1])


    

    try:
        id_imag.save(id_image_path)
        second_image.save(second_image_path)
    except Exception as e:
        return jsonify({"status": "failure", "message": f"Error saving images: {str(e)}"}), 500

    # Check if files were saved correctly
    if not os.path.exists(id_image_path) or not os.path.exists(second_image_path):
        return jsonify({"status": "failure", "message": "Error saving images."}), 500
    
    print("before :" + id_image_path)
    id_image_path_= extract_faces(id_image_path)
    print("after"+ id_image_path_)
    # id_image_path = os.path.join("face_extracted",id_image_path_ )
    # Extract embeddings for both images
    id_embedding = extract_embedding(id_image_path)
    live_image_path =os.path.join("uploads", "live_image.png")
    print(live_image_path)
    live_image_embeddings = extract_embedding(live_image_path)
    
    if id_embedding is None or live_image_embeddings is None:
        return jsonify({"status": "failure", "message": "Error in embedding extraction."}), 500

    # Compare the embeddings manually
    # is_verified = compare_embeddings(id_embedding, live_image_embeddings)
    # from deepface import DeepFace
    result = DeepFace.verify(live_image_path, id_image_path_)

    # print(f"Are the images identical? {result['verified']}")
    # print(f"Similarity score: {result['distance']}")
    is_verified = result['verified']
    # Safely remove temporary files
    try:
        if os.path.exists(id_image_path):
            os.remove(id_image_path)
        if os.path.exists(second_image_path):
            os.remove(second_image_path)
    except Exception as e:
        print(f"Error deleting files: {str(e)}")

    if is_verified:
        return jsonify({"status": "success", "message": "User identity verified."})
    else:
        return jsonify({"status": "failure", "message": "Face does not match."})
    




# Path to the FAISS index file
FAISS_INDEX_PATH = 'image_embeddings.index'

# Load the FAISS index
index = faiss.read_index(FAISS_INDEX_PATH)

# Path to dataset images (the same images used to create the FAISS index)
dataset_folder = 'dt'
image_paths = [os.path.join(dataset_folder, f) for f in os.listdir(dataset_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]

# Function to extract embedding for a query image using DeepFace
def extract_embedding(image_path):
    embedding = DeepFace.represent(img_path=image_path, model_name='Facenet', enforce_detection=False)
    return np.array(embedding[0]['embedding'])

# Ensure uploads folder exists
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Route to handle image uploads and FAISS-based similarity search
@app.route('/search', methods=['POST'])
def search_image():
    # Check if a file was uploaded
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']

    # Save the uploaded file to the server
    if file.filename == '':
        return jsonify({"error": "No filename provided"}), 400
    
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Extract embedding for the uploaded image
    query_embedding = extract_embedding(file_path).astype('float32').reshape(1, -1)

    # Search for the top K nearest neighbors in the FAISS index
    top_k = 5  # You can adjust this value
    distances, indices = index.search(query_embedding, top_k)

    # Calculate similarity scores (can be modified depending on your needs)
    similarity_scores = 1 / (1 + distances)  # Example similarity calculation

    # Map indices back to image paths
    similar_image_paths = [image_paths[idx] for idx in indices[0]]
    
    # Prepare the result
    result = [{"image_path": path, "similarity_score": float(score)} for path, score in zip(similar_image_paths, similarity_scores[0])]
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)

