from __future__ import annotations
'''
classe qui prend en entrée un retrieval et un inputDescriptor et qui s'occupe d'assembler les résultats en créant un fichier markdown qui contient les images et les paragraphes associés.
'''

from typing import TYPE_CHECKING
from pathlib import Path
if TYPE_CHECKING:
    from retrieval import Retrieval
    from inputDescriptor import InputDescriptor


class Postprocessor:
    def __init__(self, retrieval : Retrieval, inputDescriptor : InputDescriptor):
        self.retrieval = retrieval
        self.inputDescriptor = inputDescriptor
        #récupération du titre, des paragraphes et des images retrouvées correspondantes
        self.title = inputDescriptor.title
        self.paragraphs = inputDescriptor.paragraphs
        

    #fonction qui crée un fichier markdown avec les paragraphes et les images associées
    def rebuild(self):
        #construction du titre du document
        title = self.title + " - Illustrated version"

        #création et écriture du fichier markdown
        with open('results/' + title + ".md", "w", encoding="utf-8") as f:
            f.write(f"# {self.title}\n\n")
            for i in range(len(self.paragraphs)):
                f.write(f"{self.paragraphs[i]}\n")


                (image_path, dataset_name, score) = self.retrieval.getMatchingImages(i)[0] # on prend la première image parmi les top_k images les plus similaires
                image_path = Path(image_path).as_posix()  # convertit automatiquement \ en / pour que ça marche sur markdown

                f.write(f'<p align="center">\n  <img src="{image_path}" alt="Image {i} issue de {dataset_name} avec un score de similarité de {score}" />\n</p>\n')
                f.write("\n")
                
                
## Exemple d'utilisation
"""
from retrieval import Retrieval
from inputDescriptor import InputDescriptor

retrieval = Retrieval("data/paysages")
images = retrieval.images
input = InputDescriptor("textes/petite-sirene.pdf")
paragraphs = input.extractParagraphs()
postprocessor = Postprocessor(retrieval, input)
postprocessor.rebuild()
"""