from __future__ import annotations # pour éviter les problèmes de dépendances circulaires entre les classes Retrieval, InputDescriptor et DatasetDescriptor
'''
classe qui prend en entrée un inputDescriptor et un datasetDescriptor et qui s'occupe de matcher les paragraphes avec des images du dataset
'''

import torch
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

    def match(self):
        '''
        Pour chaque paragraphe on trouve les top_k images les plus similaires.
        '''
        text_embeddings = torch.stack(self.inputDescriptor.paragraphsDescriptors) # (nb_paragraphes, dim_embedding)
        image_embeddings = torch.stack(self.datasetDescriptor.imagesDescriptors) # (nb_images, dim_embedding)

        similarity = text_embeddings @ image_embeddings.T  # (nb_paragraphes, nb_images)

        self.matching_images = []
        for paragraph_index in range(similarity.shape[0]):
            scores = similarity[paragraph_index]  #(nb_images)
            top_k_indices = torch.topk(scores, k=self.top_k).indices  # indices des top_k images
    
            matches = []
            for img_index in top_k_indices:
                img_path, dataset_name = self.datasetDescriptor.imagesPaths[img_index]
                score = scores[img_index].item()
                matches.append((img_path, dataset_name, score))
            
            self.matching_images.append(matches)

        return self.matching_images
    
    def getMatchingImages(self, paragraph_index):
        '''
        Retourne les top_k images les plus similaires pour un paragraphe donné.
        '''
        if not self.matching_images:
            raise RuntimeError("Il faut appeler match() avant getMatchForParagraph().")
        return self.matching_images[paragraph_index]


