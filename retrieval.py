from __future__ import annotations # pour éviter les problèmes de dépendances circulaires entre les classes Retrieval, InputDescriptor et DatasetDescriptor
'''
classe qui prend en entrée un inputDescriptor et un datasetDescriptor et qui s'occupe de matcher les paragraphes avec des images du dataset
'''

import torch
from pathlib import Path
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from inputDescriptor import InputDescriptor
    from datasetDescriptor import DatasetDescriptor

class Retrieval:
    def __init__(self, inputDescriptor : InputDescriptor, datasetDescriptor : DatasetDescriptor, top_k : int = 5):
        self.inputDescriptor = inputDescriptor
        self.datasetDescriptor = datasetDescriptor
        self.top_k = top_k
        self.matching_images = [] # (nb_paragraphes, top_k) où matching_images[i][j] contient le tuple (image_path, dataset_name, score) pour le paragraphe i et l'image j parmi les top_k images les plus similaires


    def _prepareImageDescriptors(self):
        '''
        Selon que self.datasetDescriptor.imagesDescriptors vient d'être calculé où extrait du checkpoint, il peut être soit une liste de tenseurs 1D soit un tenseur 2D. Cette fonction s'assure que c'est toujours un tenseur 2D.
        '''
        descriptors = self.datasetDescriptor.imagesDescriptors
        if isinstance(descriptors, list):
            return torch.stack(descriptors)
        return descriptors

    def match(self, device=None):
        '''
        Pour chaque paragraphe on trouve les top_k images les plus similaires.
        '''
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"


        image_embeddings = self._prepareImageDescriptors().float().to(device) # (nb_images, dim_embedding)
        text_descriptors_list = self.inputDescriptor.paragraphsDescriptors

        self.matching_images = []
        for i in range(len(text_descriptors_list)):
            paragraph_emb = text_descriptors_list[i].float().to(device) # (1, 512) ou (nb_chunks, 512) selon la stratégie d'extraction des descripteurs de texte choisie
            
            # Calcul de la similarité cosinus entre tous les chunks du paragraphe et toutes les images
            # (nb_chunks, 512) @ (512, nb_images) -> (nb_chunks, nb_images)
            similarities = paragraph_emb @ image_embeddings.T
            
            # On prend le meilleur chunk pour chaque image : si nb_chunks > 1, on réduit la dimension 0 en gardant le maximum
            # scores shape: [nb_images]
            scores, _ = similarities.max(dim=0)
            
            print(f"Paragraph {i} - min: {scores.min():.3f}, max: {scores.max():.3f}, mean: {scores.mean():.3f}")
            
            # Récupération des top_k
            top_k_values, top_k_indices = torch.topk(scores, k=min(self.top_k, len(scores)))
            
            matches = []
            for idx, score in zip(top_k_indices, top_k_values):
                source, dataset_name = self.datasetDescriptor.imagesPaths[idx.item()]
                if isinstance(source, str):
                    # Image locale : on s'assure juste d'avoir le chemin absolu
                    final_path = Path(source).resolve().as_posix()
                else:
                    # Image HF : on demande au descriptor de la sauvegarder de dowload et de nous retourner le chemin local
                    final_path = self.datasetDescriptor._save_hf_image(source, dataset_name)
                matches.append({
                    'path': final_path,
                    'dataset': dataset_name,
                    'score': score.item(),
                    'original_source': source
                })

            self.matching_images.append(matches)
        return self.matching_images
    
    def getMatchingImages(self, paragraph_index):
        '''
        Retourne les top_k images les plus similaires pour un paragraphe donné.
        '''
        if not self.matching_images:
            raise RuntimeError("Il faut appeler match() avant getMatchForParagraph().")
        return self.matching_images[paragraph_index]


