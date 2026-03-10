import os
import uuid
import time
import base64
import warnings
import numpy as np
from PIL import Image
from io import BytesIO
from flask import Flask, jsonify, request
from flask_cors import CORS
import torch
import clip

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Paths
CHECKPOINT_PATH = "/home/all/Full_model_lr=5e-7/model_save_epoch99.pt"
EMBEDDINGS_PATH = "/home/all/all_data_embeddings.pt"
IMAGES_DIR = "/home/all/Demo"

# Load model once at startup
print("Loading CLIP model...")
model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
checkpoint = torch.load(CHECKPOINT_PATH)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print("Model loaded successfully")

# Load and normalize image embeddings once at startup
print("Loading image embeddings...")
image_embeddings = torch.load(EMBEDDINGS_PATH)
for i in range(len(image_embeddings)):
    image_embeddings[i] /= image_embeddings[i].norm(dim=-1, keepdim=True)
print(f"Loaded {len(image_embeddings)} image embeddings")

# Flask app
app = Flask(__name__)
CORS(app)

def decode_base64_image(base64_string):
    """Decode base64 string to PIL Image"""
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    image_data = base64.b64decode(base64_string)
    image = Image.open(BytesIO(image_data)).convert('RGB')
    return image

def encode_image_to_base64(image_path):
    """Encode image file to base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_similar_images(features, top_k=6):
    """Find top k similar images using cosine similarity"""
    with torch.no_grad():
        similarity = (100.0 * image_embeddings @ features.T).squeeze().cpu().numpy()
    top_indices = np.argsort(similarity)[::-1][:top_k]
    return top_indices, similarity

@app.route('/', methods=['GET'])
def index():
    return jsonify({'status': 'running'}), 200

@app.route('/test', methods=['GET'])
def test():
    return jsonify({'success': True}), 200

@app.route('/similar_images', methods=['POST'])
def similar_images():
    tic = time.time()
    trx_id = str(uuid.uuid4())

    try:
        req = request.get_json()

        if not req:
            return jsonify({
                'success': False,
                'error': 'No JSON data received'
            }), 400

        # --- Image Input ---
        if 'img' in req:
            img_data = req['img']

            # Validate base64 image
            if not (len(img_data) > 11 and img_data[0:11] == "data:image/"):
                return jsonify({
                    'success': False,
                    'error': 'Invalid image format. Must be base64 encoded string'
                }), 400

            # Decode and preprocess image
            pil_image = decode_base64_image(img_data)
            image_tensor = preprocess(pil_image).unsqueeze(0).to(device)

            # Extract image features
            with torch.no_grad():
                features = model.encode_image(image_tensor)
            features /= features.norm(dim=-1, keepdim=True)

        # --- Text Input ---
        elif 'text' in req:
            text_query = req['text']

            if not text_query or len(text_query.strip()) == 0:
                return jsonify({
                    'success': False,
                    'error': 'Text query cannot be empty'
                }), 400

            # Tokenize and encode text
            text_tokens = clip.tokenize([text_query]).to(device)
            with torch.no_grad():
                features = model.encode_text(text_tokens)
            features /= features.norm(dim=-1, keepdim=True)

        else:
            return jsonify({
                'success': False,
                'error': 'Request must contain either img or text field'
            }), 400

        # Get top 6 similar images
        top_indices, similarity_scores = get_similar_images(features, top_k=6)

        # Load and encode similar images as base64
        similar_images_base64 = []
        for idx in top_indices:
            image_filename = f'p ({idx + 1}).jpg'
            image_path = os.path.join(IMAGES_DIR, image_filename)
            print(f"Image: {image_filename} | Similarity: {similarity_scores[idx]:.2f}")

            if os.path.exists(image_path):
                encoded_image = encode_image_to_base64(image_path)
                similar_images_base64.append(encoded_image)

        toc = time.time()

        return jsonify({
            'success': True,
            'img': similar_images_base64,
            'count': len(similar_images_base64),
            'trx_id': trx_id,
            'seconds': round(toc - tic, 3)
        }), 200

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'trx_id': trx_id
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)