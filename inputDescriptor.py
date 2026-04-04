'''
classe qui prend en entrée un fichier texte raw et qui s'occupe d'extraire les paragraphes et des descripteurs de ces paragraphes. 
'''
from PyPDF2 import PdfReader #lecture de pdf
import torch
from transformers import pipeline

class InputDescriptor:
    def __init__(self, inputPath : str):
        self.inputPath = inputPath
        self.paragraphs = []
        self.paragraphsDescriptors = []
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
                self.title = self.inputPath.split("/")[-1][:-4]
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

    def extractDescriptors(self, paragraphs, processor, model, device, strategy="sliding_window"):
        '''
        input :
        - paragraphs : liste de paragraphes à projeter dans l'espace latent de CLIP
        - processor : le processor de CLIP
        - model : le modèle CLIP
        - device : le device sur lequel faire les calculs (cpu ou gpu)
        - strategy : la stratégie à adopter pour projeter les paragraphes dans l'espace latent de CLIP. 
            - "sliding_window" : on divise les paragraphes en morceaux de 77 tokens (taille maximale d'entrée pour CLIP) et on projette chaque morceau séparément on fait ensuite la moyenne. Remarque : on fait overlap de 10 tokens entre les morceaux pour éviter de couper des phrases en deux.
            - "llm" : on utilise un modèle de langage pour reformuler les paragraphes en des phrases plus courtes qui contiennent l'essentiel de l'information du paragraphe. Cette stratégie est plus coûteuse en temps de calcul mais elle permet d'obtenir de meilleurs résultats car on perd moins d'information.
            - "truncate" : on tronque les paragraphes pour les faire rentrer dans CLIP : cette stratégie est plus rapide mais aussi très naive car on perd beaucoup d'information dès que les paragraphes sont un peu longs.
        '''

        if strategy == "llm":        
            captioner = pipeline("text-generation", model="Qwen/Qwen2.5-1.5B-Instruct", device=0)


        descriptors = []
        for p in paragraphs:
            if isinstance(p, str):

                if strategy == "truncate":
                    inputs = processor(text=[p], return_tensors="pt", padding=True, truncation=True, max_length=77).to(device)
                    with torch.no_grad():
                        features = model.get_text_features(**inputs).pooler_output
                        features /= features.norm(p=2, dim=-1, keepdim=True)
                        descriptors.append(features.squeeze())

                elif strategy == "sliding_window":
                    token_ids = processor.tokenizer(p, truncation=False, return_tensors="pt")['input_ids'][0]

                    if len(token_ids) <= 77:  # pas besoin de sliding window
                        inputs = processor(text=[p], return_tensors="pt", padding=True, truncation=True, max_length=77).to(device)
                        with torch.no_grad():
                            features = model.get_text_features(**inputs).pooler_output
                            features /= features.norm(p=2, dim=-1, keepdim=True)
                            descriptors.append(features.squeeze())

                    else:
                        chunk_size = 77
                        overlap = 10
                        chunks = [token_ids[i:i+chunk_size] for i in range(0, len(token_ids), chunk_size - overlap)]
                        
                        descriptors_chunks = []
                        for chunk in chunks:
                            pad_length = chunk_size - len(chunk)
                            if pad_length > 0:
                                chunk = torch.cat([chunk, torch.zeros(pad_length, dtype=torch.long)])
                            input_ids = chunk.unsqueeze(0).to(device)  # [1, 77]
                            with torch.no_grad():
                                features = model.get_text_features(input_ids=input_ids).pooler_output
                                features /= features.norm(p=2, dim=-1, keepdim=True)
                                descriptors_chunks.append(features.squeeze())

                        
                        mean_emb = torch.stack(descriptors_chunks).mean(dim=0)
                        mean_emb /= mean_emb.norm(p=2, dim=-1, keepdim=True)  # on renormalise après la moyenne
                        descriptors.append(mean_emb)
                
                elif strategy == "llm":
                    prompt = f"Describe the visual scene as a short English image caption (max 15 words):\n\n{p}\n\nCaption:"
                    result = captioner(prompt, max_new_tokens=30, do_sample=False)
                    caption = result[0]['generated_text'].split("Caption:")[-1].strip()
                    inputs = processor(text=[caption], return_tensors="pt", padding=True, truncation=True, max_length=77).to(device)
                    print(f"Original paragraph: {p}")
                    print(f"Generated caption: {caption}")
                    print("____________________________________")
                    with torch.no_grad():
                        features = model.get_text_features(**inputs).pooler_output
                        features /= features.norm(p=2, dim=-1, keepdim=True)
                        descriptors.append(features.squeeze())



                else:
                    raise ValueError("Unsupported strategy. Only 'sliding_window', 'llm' and 'truncate' are supported.")

            else:
                raise Exception("The paragraph must be a text string.")

        self.paragraphsDescriptors = descriptors
        return self.paragraphsDescriptors

#test
if __name__ == "__main__":
    inputDescriptor = InputDescriptor("textes/petite-sirene.pdf")
    paragraphs = inputDescriptor.extractParagraphs()
    for i in range(len(paragraphs)):
        print(f"Paragraph {i} : {paragraphs[i]}")
        print('-----------------------------')