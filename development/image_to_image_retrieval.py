from PIL import Image
import torch
import clip
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model(checkpoint_path):
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, preprocess

def load_embeddings(embeddings_path):
    image_list = torch.load(embeddings_path)
    # Normalize embeddings
    for i in range(len(image_list)):
        image_list[i] /= image_list[i].norm(dim=-1, keepdim=True)
    return image_list

def image_to_image(query_image_path, model, preprocess, image_embeddings, top_k=6):
    # Preprocess and encode query image
    image = preprocess(Image.open(query_image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        query_features = model.encode_image(image)
    query_features /= query_features.norm(dim=-1, keepdim=True)

    # Compute cosine similarity scores
    with torch.no_grad():
        similarity = (100.0 * image_embeddings @ query_features.T).squeeze().cpu().numpy()

    # Get top k results
    top_indices = np.argsort(similarity)[::-1][:top_k]

    print(f"\nTop {top_k} results for query image: '{query_image_path}'")
    for i, idx in enumerate(top_indices):
        print(f"Rank {i+1} | Image index: {idx} | Similarity: {similarity[idx]:.2f}")

    return top_indices, similarity

if __name__ == "__main__":
    # Paths
    checkpoint_path = "/home/all/Full_model_lr=5e-7/model_save_epoch99.pt"
    embeddings_path = "/home/all/images_embeddings.pt"

    # Load model and embeddings
    model, preprocess = load_model(checkpoint_path)
    image_embeddings = load_embeddings(embeddings_path)

    # Example query
    query_image_path = "/home/all/Demo/sample.jpg"
    top_indices, similarity = image_to_image(query_image_path, model, preprocess, image_embeddings)