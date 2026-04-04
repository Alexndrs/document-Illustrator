'''
Classe qui assemble les images issues de plusieurs dataset et calcule les descripteurs de ces images. 
'''

import os
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm



class DatasetDescriptor:
    def __init__(self, datasetPath : str, processor=None):
        self.datasetPath = datasetPath
        self.imagesPaths = []
        self.imagesDescriptors = []
        self.processor = processor
    
    def getImagesPaths(self):
        '''
        Parcourt récursivement le dossier datasetPath et retourne une liste de tous les chemins d'images trouvés.
        '''
        self.dataset_names = os.listdir(self.datasetPath)
        for dataset_name in self.dataset_names:
            for root, dirs, files in os.walk(os.path.join(self.datasetPath,dataset_name)):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                        self.imagesPaths.append((os.path.join(root, file),dataset_name))
        return self.imagesPaths

    def __len__(self):
        '''
        Retourne le nombre total d'images dans tous les datasets.
        '''
        return len(self.imagesPaths)

    def showRandomImage(self):
        '''
        Affiche une image aléatoire de chaque dataset.
        '''
        for ds, paths in self.imagesPaths.items():
            if paths:
                random_image_path = random.choice(paths)
                image = Image.open(random_image_path)
                plt.imshow(image)
                plt.title(f"Random image from {ds}")
                plt.axis('off')
                plt.show()
    
    def __getitem__(self, index: int):
        '''
        Retourne une paire (image, dataset_name) pour l'image à l'index donné.
        '''
        img_path, dataset_name = self.imagesPaths[index]
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))

        # on applique le preprocessing CLIP directement ici pour éviter de le faire dans le dataloader qui ne peut pas gérer les images PIL
        pixel_values = self.processor(images=img, return_tensors="pt").pixel_values.squeeze(0)  # [3, 224, 224]
        return pixel_values, dataset_name



    def projectDatasetWithClip(self, model, batch_size : int, device : str, save_path: str = None):
        '''
        projection du dataset dans un espace latent avec CLIP.
        Si save_path est fourni, sauvegarde (ou charge depuis le cache) les projections.
        '''
        #chargement du dataset
        if save_path and os.path.exists(save_path):
            print(f"Chargement des projections depuis {save_path}")
            checkpoint = torch.load(save_path, map_location=device)
            self.imagesPaths = checkpoint['image_paths']
            self.imagesDescriptors = checkpoint['projections']
            return checkpoint['projections'], checkpoint['image_paths']


        dataloader = DataLoader(self, batch_size=batch_size, drop_last=False, shuffle=False, num_workers=0)
        projected_dataset = torch.zeros(len(self), 512) # création de la matrice de projection finale
        with torch.no_grad():
            for i, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc="Calcul des descripteurs CLIP"):
                pixel_values = batch[0].to(device)  # [B, 3, 224, 224] float, sur GPU
                features = model.get_image_features(pixel_values=pixel_values).pooler_output
                features /= features.norm(p=2, dim=-1, keepdim=True)
                projected_dataset[i*batch_size:(i+1)*batch_size] = features.cpu()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({
                'projections': projected_dataset.cpu(),
                'image_paths': self.imagesPaths,
            }, save_path)
            print(f"Projections sauvegardées dans {save_path}")
        
        self.imagesDescriptors = projected_dataset
        return self.imagesDescriptors, self.imagesPaths
