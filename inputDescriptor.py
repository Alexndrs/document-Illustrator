'''
classe qui prend en entrée un fichier texte raw et qui s'occupe d'extraire les paragraphes et des descripteurs de ces paragraphes. 
'''
from PyPDF2 import PdfReader #lecture de pdf

class InputDescriptor:
    def __init__(self, inputPath : str):
        self.inputPath = inputPath
        self.paragraphs = []
        self.title = ""
    
    def extractParagraphs(self):
        #test fichier pdf ou txt
        if self.inputPath.endswith(".pdf"):
            #extraction du titre
            title = self.inputPath.split("/")[-1]
            self.title = title[:-4]
            #extraction des paragraphes du pdf
            reader = PdfReader(self.inputPath)
            paragraphs = []
            for page in reader.pages:
                text = page.extract_text()
                paragraphs += text.split("\n\n")
        elif self.inputPath.endswith(".txt"):
            #extraction des paragraphes du txt
            with open(self.inputPath, "r") as f:
                text = f.read()
                paragraphs = text.split("\n\n")
        else:  #si le format n'est pas supporté
            raise ValueError("Unsupported file format. Only .pdf and .txt are supported.")

        if len(paragraphs) == 0:
            raise ValueError("No paragraphs found in the input file.")
        
        #on retire les paragraphes vides
        paragraphs = [p.strip() for p in paragraphs if p.strip() != '']
        
        #si le texte n'a pas de paragraphe, on le divise en morceaux de 400 caractères
        if len(paragraphs) <= 2 :
            for i in range(len(paragraphs)):
                if len(paragraphs[i]) > 400:  #vérification de la longueur du paragraphe
                    paragraphs[i] = [paragraphs[i][j:j+400] for j in range(0, len(paragraphs[i]), 400)]
                    #aplatissement de la liste de paragraphes
                    paragraphs = [item for sublist in paragraphs for item in sublist]
        self.paragraphs = paragraphs
        return paragraphs

    def extractDescriptors(self, paragraphs, processor, model, device):
        #paragraphs peut soit être une liste de paragraphes, soit une liste de phrases (générés via un llm) correspondant aux paragraphes
        #cette fonction retourne le vecteur de représentation de chaque paragraphe/phrase dans l'espace latent de CLIP
        descriptors = []
        for p in paragraphs:
            if isinstance(p, str):
                inputs = processor(text=[p], return_tensors="pt", padding=True).to(device)
                outputs = model.get_text_features(**inputs)
                outputs /= outputs.norm(p=2, dim=-1, keepdim=True)
                descriptors.append(outputs.squeeze())
            else:
                raise Exception("The paragraph must be a text string.")

        return descriptors

#test
if __name__ == "__main__":
    inputDescriptor = InputDescriptor("textes/petite-sirene.pdf")
    paragraphs = inputDescriptor.extractParagraphs()
    for i in range(len(paragraphs)):
        print(f"Paragraph {i} : {paragraphs[i]}")
        print('-----------------------------')