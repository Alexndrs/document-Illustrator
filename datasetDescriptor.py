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
from torch.utils.data import Subset
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



    def projectDatasetWithClip(self, model, batch_size : int, device : str, save_path: str = None, save_every: int = 50):
        '''
        projection du dataset dans un espace latent avec CLIP.
        Si save_path est fourni, sauvegarde (ou charge depuis le cache) les projections et calcule les projections pour les images qui ne sont pas dans le checkpoint
        '''

        # on retient les images déjà projetées pour éviter de les recalculer si elles sont déjà dans le checkpoint
        existing_data = {}

        #chargement du dataset
        if save_path and os.path.exists(save_path):
            print(f"Chargement des projections depuis {save_path}")
            checkpoint = torch.load(save_path, map_location=device)
            paths_only = [p[0] for p in checkpoint['image_paths']]
            existing_data = dict(zip(paths_only, checkpoint['projections']))
            print(f"Nombre d'images déjà indexées : {len(existing_data)}")

        missing_indices = [i for i, (path, _) in enumerate(self.imagesPaths) if path not in existing_data]
        if not missing_indices:
            print("Toutes les images sont déjà à jour dans le checkpoint.")
            self.imagesDescriptors = torch.stack([existing_data[p[0]] for p in self.imagesPaths])
            return self.imagesDescriptors, self.imagesPaths
        

        print(f"Nouvelles images à traiter : {len(missing_indices)}")
        missing_subset = Subset(self, missing_indices)
        dataloader = DataLoader(missing_subset, batch_size=batch_size, shuffle=False, num_workers=0)

        model.eval()
        new_descriptors = {}
        with torch.no_grad():
            for i, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc="Calcul des descripteurs CLIP"):
                pixel_values = batch[0].to(device)  # [B, 3, 224, 224] float, sur GPU
                features = model.get_image_features(pixel_values=pixel_values).pooler_output
                features /= features.norm(p=2, dim=-1, keepdim=True)
                features = features.cpu()

                start_idx = i * batch_size
                for j in range(features.shape[0]):
                    global_idx = missing_indices[start_idx + j]
                    img_path = self.imagesPaths[global_idx][0]
                    new_descriptors[img_path] = features[j]
                
                if save_path and (i + 1) % save_every == 0:
                    self._save_checkpoint(save_path, existing_data, new_descriptors)
                    print(f"\n[Checkpoint] Sauvegarde intermédiaire effectuée à l'étape {i+1}")

        existing_data.update(new_descriptors)
        if save_path:
            self._save_checkpoint(save_path, existing_data, {})
            print(f"Sauvegarde finale terminée dans {save_path}")

        self.imagesDescriptors = torch.stack([existing_data[p[0]] for p in self.imagesPaths])
        return self.imagesDescriptors, self.imagesPaths
    

    def _save_checkpoint(self, save_path, existing_dict, new_dict):
        ''' Fonction utilitaire pour compiler et sauvegarder le dictionnaire en listes '''
        combined = {**existing_dict, **new_dict}
        save_paths = []
        save_projs = []
        
        for path_tuple in self.imagesPaths:
            path = path_tuple[0]
            if path in combined:
                save_paths.append(path_tuple)
                save_projs.append(combined[path])
        
        if save_projs:
            torch.save({
                'projections': torch.stack(save_projs).cpu(),
                'image_paths': save_paths,
            }, save_path)