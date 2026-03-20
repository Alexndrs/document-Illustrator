'''
classe qui prend en entrée un fichier texte raw et qui s'occupe d'extraire les paragraphes et des descripteurs de ces paragraphes. 
'''
from PyPDF2 import PdfReader #lecture de pdf

class InputDescriptor:
    def __init__(self, inputPath : str):
        self.inputPath = inputPath
    
    def extractParagraphs(self):
        #test fichier pdf ou txt
        if self.inputPath.endswith(".pdf"):
            #extraire les paragraphes du pdf
            reader = PdfReader(self.inputPath)
            paragraphs = []
            for page in reader.pages:
                text = page.extract_text()
                paragraphs += text.split("\n\n")
        elif self.inputPath.endswith(".txt"):
            #extraire les paragraphes du txt
            with open(self.inputPath, "r") as f:
                text = f.read()
                paragraphs = text.split("\n\n")
        else:
            raise ValueError("Unsupported file format. Only .pdf and .txt are supported.")

        return paragraphs


#test
if __name__ == "__main__":
    inputDescriptor = InputDescriptor("textes/petite-sirene.pdf")
    paragraphs = inputDescriptor.extractParagraphs()
    for i in range(len(paragraphs)):
        print(f"Paragraph {i} : {paragraphs[i]}")
        print('-----------------------------')
