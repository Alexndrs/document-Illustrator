'''
Classe qui assemble les images issues de plusieurs dataset et calcule les descripteurs de ces images. 
'''

import os
import random
from pathlib import Path
from typing import List
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from tqdm import tqdm
from datasets import load_dataset



class DatasetDescriptor:
    def __init__(self, datasetPath : str, hf_dataset_names : List[str] = [], processor=None):
        self.datasetPath = datasetPath
        self.hf_dataset_names = hf_dataset_names
        self.processor = processor

        self.imagesPaths = [] # (source, dataset_name) avec source = chemin str si local ou int (index huggingface) si dataset HuggingFace
        self.imagesDescriptors = []
        self.hf_datasets = {} # { hf_dataset_name: hf_dataset_object }
    
    def getImagesPaths(self):
        '''
        Parcourt récursivement le dossier datasetPath et retourne une liste de tous les chemins d'images trouvés.
        Et load les datasets huggingfaces spécifiés
        '''
        self.dataset_names = os.listdir(self.datasetPath)
        for dataset_name in self.dataset_names:
            for root, dirs, files in os.walk(os.path.join(self.datasetPath,dataset_name)):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                        self.imagesPaths.append((os.path.join(root, file),dataset_name))

        for hf_name in self.hf_dataset_names:
            print(f"Chargement du dataset hugging face : {hf_name}")
            ds = load_dataset(hf_name, split="train")
            self.hf_datasets[hf_name] = ds
            dataset_name = hf_name.split("/")[-1]
            for i in range(len(ds)):
                self.imagesPaths.append(((hf_name, i), dataset_name))

        print(f"Total : {len(self.imagesPaths)} images (dont {len(self.hf_datasets)} datasets HF)")
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
        source, dataset_name = self.imagesPaths[index]

        try:
            if isinstance(source, str):
                img = Image.open(source).convert("RGB")
            else:
                # image d'un hugging face dataset
                hf_name, hf_index = source
                row = self.hf_datasets[hf_name][hf_index]
                img = row.get("image") or row.get("IMAGE")
                if img is None:
                    raise ValueError(f"Champ 'image' (ou 'IMAGE') introuvable dans {hf_name}")
                if not isinstance(img, Image.Image):
                    img = Image.open(img).convert("RGB")
                else:
                    img = img.convert("RGB")

        except Exception:
            img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))

        return img, dataset_name

    def _save_hf_image(self, hf_source, dataset_name):
        '''
        hf_source : tuple (hf_name, hf_index) qui identifie de manière unique une image dans les datasets hugging face

        Enregistre localement une image hugging face et retourne son chemin d'accès local
        '''
        hf_name, hf_index = hf_source
        
        # Création du sous-dossier spécifique au dataset
        safe_ds_name = dataset_name.replace("/", "_") # On remplace les / par _ pour éviter des problèmes de dossiers imbriqués
        ds_dir = Path("./data") / f"{safe_ds_name}-dl"
        ds_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = ds_dir / f"image_{hf_index}.jpg"
        
        # On n'extrait l'image que si elle n'existe pas déjà
        if not file_path.exists():
            # On cherche l'index global dans DatasetDescriptor pour utiliser __getitem__
            if hf_name not in self.hf_datasets:
                raise KeyError(f"Le dataset HF '{hf_name}' n'est pas chargé.")
                
            row = self.hf_datasets[hf_name][hf_index]
            img_data = row.get("image") or row.get("IMAGE")
            
            if img_data is None:
                # Création d'une image vide par défaut en cas d'erreur de source
                img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
            elif not isinstance(img_data, Image.Image):
                img = Image.open(img_data).convert("RGB")
            else:
                img = img_data.convert("RGB")
            
            img.save(file_path, "JPEG")
        
        return file_path.resolve().as_posix()
        

    def _unified_key(self, source) -> str:
        '''
        On unifie les sources en créant une clé qui est simplement une str pour simplifier le code en limitant les disjonctions de cas entre path local et path hugging face
        e.g : les clés seront de la forme :
        
        local  → "/path/to/img.jpg"
        hf     → "('fantasyfish/laion-art', 42)"

        '''
        return str(source)

    def projectDatasetWithClip(self, model, batch_size: int, device: str,
                               save_path: str = None, save_every: int = 50):
        existing_data = {}

        if save_path and os.path.exists(save_path):
            print(f"Chargement checkpoint : {save_path}")
            checkpoint = torch.load(save_path, map_location="cpu")
            keys = [self._unified_key(p[0]) for p in checkpoint['image_paths']]
            existing_data = dict(zip(keys, checkpoint['projections']))
            print(f"Images déjà indexées : {len(existing_data)}")

        missing_indices = [
            i for i, (source, _) in enumerate(self.imagesPaths)
            if self._unified_key(source) not in existing_data
        ]

        if not missing_indices:
            print("Checkpoint à jour.")
            self.imagesDescriptors = torch.stack(
                [existing_data[self._unified_key(p[0])] for p in self.imagesPaths]
            )
            return self.imagesDescriptors, self.imagesPaths

        print(f"Images à traiter : {len(missing_indices)}")

        def collate_fn(batch):
            # cette fonction dit au DataLoader de ne pas essayer de transformer les images PIL en tenseurs tout de suite
            return [item[0] for item in batch], [item[1] for item in batch]
        
        dataloader = DataLoader(Subset(self, missing_indices), batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)

        model.eval()
        new_descriptors = {}
        with torch.no_grad():
            for i, (images, names) in tqdm(enumerate(dataloader), total=len(dataloader), desc="Descripteurs CLIP"):
                
                inputs = self.processor(images=images, return_tensors="pt").to(device)
                outputs = model.vision_model(**inputs)
                pooled = outputs.pooler_output
                features = model.visual_projection(pooled)

                features = features / features.norm(p=2, dim=-1, keepdim=True)
                features = features.cpu()


                for j in range(features.shape[0]):
                    global_idx = missing_indices[i * batch_size + j]
                    key = self._unified_key(self.imagesPaths[global_idx][0])
                    new_descriptors[key] = features[j]

                if save_path and (i + 1) % save_every == 0:
                    self._save_checkpoint(save_path, existing_data, new_descriptors)

        existing_data.update(new_descriptors)
        if save_path:
            self._save_checkpoint(save_path, existing_data, {})

        self.imagesDescriptors = torch.stack(
            [existing_data[self._unified_key(p[0])] for p in self.imagesPaths]
        )
        return self.imagesDescriptors, self.imagesPaths

    def _save_checkpoint(self, save_path, existing_dict, new_dict):
        combined = {**existing_dict, **new_dict}
        save_paths, save_projs = [], []
        for path_tuple in self.imagesPaths:
            key = self._unified_key(path_tuple[0])
            if key in combined:
                save_paths.append(path_tuple)
                save_projs.append(combined[key])
        if save_projs:
            torch.save({
                'projections': torch.stack(save_projs).cpu(),
                'image_paths': save_paths,
            }, save_path)