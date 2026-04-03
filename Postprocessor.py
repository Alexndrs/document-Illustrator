'''
classe qui prend en entrée un retrieval et un inputDescriptor et qui s'occupe d'assembler les résultats en créant un fichier markdown qui contient les images et les paragraphes associés.
'''


class Postprocessor:
    def __init__(self, retrieval, inputDescriptor):
        self.retrieval = retrieval
        self.inputDescriptor = inputDescriptor
        #récupération du titre, des paragraphes et des images retrouvées correspondantes
        self.title = inputDescriptor.title
        self.paragraphs = inputDescriptor.paragraphs
        self.images = retrieval.images
        

    #fonction qui crée un fichier markdown avec les paragraphes et les images associées
    def rebuild(self):
        #construction du titre du document
        title = self.title + " - Illustrated version"

        #création et écriture du fichier markdown
        with open(title + ".md", "w") as f:
            f.write(f"# {self.title}\n\n")
            for i in range(len(self.paragraphs)):
                f.write(f"{self.paragraphs[i]}\n")
                #sécurité pour éviter les erreurs d'index si il y a moins d'images que de paragraphes
                if i < len(self.images):
                    f.write(f'<p align="center">\n  <img src="{self.images[i]}" alt="Image {i}" />\n</p>\n')
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