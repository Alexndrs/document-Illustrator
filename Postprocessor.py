from __future__ import annotations
'''
classe qui prend en entrée un retrieval et un inputDescriptor et qui s'occupe d'assembler les résultats en créant un fichier markdown qui contient les images et les paragraphes associés.
'''

from typing import TYPE_CHECKING
from pathlib import Path
from PIL import Image
if TYPE_CHECKING:
    from retrieval import Retrieval
    from inputDescriptor import InputDescriptor


class Postprocessor:
    def __init__(self, retrieval : Retrieval, inputDescriptor : InputDescriptor, output_path : str):
        self.retrieval = retrieval
        self.inputDescriptor = inputDescriptor
        #récupération du titre, des paragraphes et des images retrouvées correspondantes
        self.title = inputDescriptor.title
        self.paragraphs = inputDescriptor.paragraphs
        self.output_path = output_path

    #fonction qui crée un fichier markdown avec les paragraphes et les images associées
    def rebuild(self):
        #construction du titre du document
        title = self.title + " - Illustrated version"

        #création et écriture du fichier markdown
        with open(self.output_path, "w", encoding="utf-8") as f:
            f.write(f"# {title}\n\n")
            for i in range(len(self.paragraphs)):
                f.write(f"{self.paragraphs[i]}\n")


                best_match = self.retrieval.getMatchingImages(i)[0] # on prend la première image parmi les top_k images les plus similaires
                path, dataset_name, score = best_match['path'],best_match['dataset'],best_match['score']

                img_tag = f'<img src="{path}" alt="Image {i} ({dataset_name})" width="500" />'
                caption = f'<br/><em style="font-size:0.8em">Source: {dataset_name} | Score: {score:.3f}</em>'
                
                f.write(f'<p align="center">\n  {img_tag}\n  {caption}\n</p>\n\n')