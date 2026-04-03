"""
filter_images.py
Garde N images dans un dossier en supprimant les trop similaires.
Utilise CLIP pour les embeddings et faiss pour la recherche vectorielle rapide.

Usage:
    python datasetFilterer.py --folder ./Romanticism --n 500
"""

import argparse
from pathlib import Path

import faiss
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


def load_model(device):
    print("Chargement de CLIP...")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    model.eval()
    return processor, model


def compute_embeddings(image_paths, processor, model, device, batch_size):
    """Calcule les embeddings CLIP pour une liste d'images, par batch."""
    all_embeddings = []
    valid_paths = []
    failed = []

    for i in tqdm(range(0, len(image_paths), batch_size), desc="Embeddings"):
        batch_paths = image_paths[i : i + batch_size]
        images = []
        batch_valid = []

        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                images.append(img)
                batch_valid.append(p)
            except Exception:
                failed.append(p)

        if not images:
            continue

        inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            features = model.vision_model(pixel_values=inputs["pixel_values"]).pooler_output
            features = model.visual_projection(features)
            features = features / features.norm(dim=-1, keepdim=True)


        all_embeddings.append(features.cpu().numpy())
        valid_paths.extend(batch_valid)

    if failed:
        print(f"  {len(failed)} images ignorées (illisibles)")

    return valid_paths, np.vstack(all_embeddings).astype("float32")


def score_images(paths):
    """Résolution × aspect ratio — on préfère les grandes images bien proportionnées."""
    scores = []
    for p in paths:
        try:
            img = Image.open(p)
            w, h = img.size
            scores.append(w * h * min(w, h) / max(w, h))
        except Exception:
            scores.append(0.0)
    return np.array(scores)


def deduplicate_faiss(sorted_paths, sorted_embs, n, threshold):
    """
    Déduplication greedy avec faiss IndexFlatIP (faiss est optimisé pour la recherche vectorielle et donc plus rapide)
     
    produit scalaire = dist cosinus sur le vecteur d'embedding normalisé

    Pour chaque image (triée par qualité décroissante) :
      - on cherche si une image déjà gardée est trop similaire
      - si non, on l'ajoute à l'index faiss
    """
    dim = sorted_embs.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner Product = cosine sur vecteurs L2-normalisés

    kept = []

    for path, emb in tqdm(
        zip(sorted_paths, sorted_embs),
        total=len(sorted_paths),
        desc="Déduplication"
    ):
        if index.ntotal > 0:
            emb_query = emb.reshape(1, -1)
            distances, _ = index.search(emb_query, k=1)
            if distances[0][0] >= threshold:
                continue  # trop similaire à une image déjà gardée

        index.add(emb.reshape(1, -1))
        kept.append(path)

        if len(kept) >= n:
            break

    return kept


def filter_images(folder, n, threshold, batch_size):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device}")

    extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    all_images = [p for p in folder.rglob("*") if p.suffix.lower() in extensions]
    print(f"Images trouvées : {len(all_images)}")

    if len(all_images) <= n:
        print(f"Déjà {len(all_images)} images (≤ {n}), rien à faire.")
        return

    processor, model = load_model(device)

    # 1. Embeddings
    valid_paths, embeddings = compute_embeddings(
        all_images, processor, model, device, batch_size
    )
    print(f"Embeddings calculés : {len(valid_paths)} images")

    # 2. Tri par qualité décroissante
    print("Calcul des scores de qualité...")
    scores = score_images(valid_paths)
    order = np.argsort(scores)[::-1]
    sorted_paths = [valid_paths[i] for i in order]
    sorted_embs = embeddings[order]

    # 3. Déduplication via faiss
    kept = deduplicate_faiss(sorted_paths, sorted_embs, n, threshold)

    kept_set = set(kept)
    to_delete = [p for p in valid_paths if p not in kept_set]

    print(f"\nRésultat :")
    print(f"  Images gardées    : {len(kept_set)}")
    print(f"  Images supprimées : {len(to_delete)}")

    print("\nSuppression en cours...")
    for p in tqdm(to_delete):
        p.unlink()
    print("Terminé.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Filtre un dossier d'images par similarité CLIP + faiss."
    )
    parser.add_argument("--folder",     type=str,   required=True)
    parser.add_argument("--n",          type=int,   required=True,
                        help="Nombre d'images à garder")
    parser.add_argument("--threshold",  type=float, default=0.95,
                        help="Seuil similarité cosine (0-1). Défaut: 0.95")
    parser.add_argument("--batch-size", type=int,   default=64,
                        help="Taille des batchs CLIP. Défaut: 64")
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.exists():
        print(f"Erreur : le dossier '{folder}' n'existe pas.")
        exit(1)

    filter_images(folder, args.n, args.threshold, args.batch_size)