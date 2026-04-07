'''
classe qui prend en entrée un fichier texte raw et qui s'occupe d'extraire les paragraphes et des descripteurs de ces paragraphes. 
'''
from PyPDF2 import PdfReader #lecture de pdf
import torch
from transformers import pipeline
from logger import save_logs

class InputDescriptor:
    def __init__(self, inputPath : str,):
        self.inputPath = inputPath
        self.paragraphs = []
        self.paragraphsDescriptors = []
        self.title = ""
        
    def extractParagraphs(self):
        if self.inputPath.endswith(".pdf"):
            title = self.inputPath.split("/")[-1]
            self.title = title[:-4]
            reader = PdfReader(self.inputPath)
            paragraphs = []
            for page in reader.pages:
                text = page.extract_text()
                paragraphs += text.split("\n\n")
        elif self.inputPath.endswith(".txt"):
            #extraction des paragraphes du txt
            with open(self.inputPath, "r") as f:
                self.title = self.inputPath.split("/")[-1][:-4]
                try:
                    text = f.read()
                    paragraphs = text.split("\n\n")
                except Exception as e:
                    save_logs(f"Error while reading the input file: {e}")
                    return []
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

    def extractDescriptors(self, paragraphs, processor, model, device, strategy="best_matching_chunk"):
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
            - "best_matching_chunk" : 
            - "keywords"
        '''

        if strategy == "llm":        
            captioner = pipeline("text-generation", model="Qwen/Qwen2.5-1.5B-Instruct", device=0, torch_dtype=torch.float16)
        
        if strategy == "keywords":
            captioner = pipeline("text-generation", model="Qwen/Qwen2.5-1.5B-Instruct", device=0, torch_dtype=torch.float16)


        descriptors = []
        for p in paragraphs:
            if not isinstance(p, str):
                continue

            if strategy in ["truncate", "llm", "keywords"]:
                # avec ces stratégies, on obtient un seul vecteur par paragraphe, donc on peut faire la projection directement
                if strategy == "truncate":
                    text_input = p
                elif strategy == "llm":
                    prompt = f"<|im_start|>system\nYou are an image captioning assistant. Given a text, output ONLY a short English visual scene description in 10 words maximum. Focus on what is visually present: characters, setting, objects. No story, no narrative.<|im_end|>\n<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\nVisual scene:"
                    caption = captioner(prompt, max_new_tokens=20,max_length=None, do_sample=False)
                    text_input = caption[0]['generated_text'].split("Visual scene:")[-1].strip()
                    text_input = text_input.split("\n")[0].strip()
                    save_logs(f"Original paragraph: {p}\nLLM description: {text_input}\n")
                elif strategy == "keywords":
                    prompt = f"<|im_start|>system\nList 5 English visual keywords for this scene, separated by commas.<|im_end|>\n<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n"
                    keywords = captioner(prompt, max_new_tokens=20,max_length=None, do_sample=False)
                    text_input = keywords[0]['generated_text'].split("assistant\n")[-1].strip()
                    save_logs(f"Original paragraph: {p}\nExtracted keywords: {text_input}\n")

                inputs = processor(text=[text_input], return_tensors="pt", padding=True, truncation=True, max_length=77).to(device)
                with torch.no_grad():
                    features = model.get_text_features(**inputs).pooler_output
                    features /= features.norm(p=2, dim=-1, keepdim=True)
                    descriptors.append(features) # (1, 512) (car on a un seul vecteur par paragraphe avec ces stratégies)

            elif strategy in ["sliding_window", "best_matching_chunk"]:
                # avec ces stratégies, on obtient plusieurs vecteurs par paragraphe, donc on doit d'abord faire la projection puis éventuellement faire une agrégation (moyenne ou max) pour rester compatible avec le reste du code qui suppose qu'on a un seul vecteur par paragraphe.
                token_ids = processor.tokenizer(p, truncation=False, return_tensors="pt")['input_ids'][0]
                chunk_size = 77
                overlap = 10
                chunks = [token_ids[i:i+chunk_size] for i in range(0, len(token_ids), chunk_size - overlap)]
                
                chunk_features = []
                for chunk in chunks:
                    if len(chunk) < chunk_size:
                        chunk = torch.cat([chunk, torch.zeros(chunk_size - len(chunk), dtype=torch.long)])
                    input_ids = chunk.unsqueeze(0).to(device)
                    with torch.no_grad():
                        features = model.get_text_features(input_ids=input_ids).pooler_output
                        features /= features.norm(p=2, dim=-1, keepdim=True)
                        chunk_features.append(features) # (1, 512)
                
                combined_chunks = torch.cat(chunk_features, dim=0) # (nb_chunks, 512)

                if strategy == "sliding_window":
                    # On fait la moyenne
                    mean_feat = combined_chunks.mean(dim=0, keepdim=True)
                    mean_feat /= mean_feat.norm(p=2, dim=-1, keepdim=True)
                    descriptors.append(mean_feat) # (1, 512)
                else:
                    # strategy == "best_matching_chunk"
                    # On garde la matrice de tous les chunks et dans retrieval on prendra le chunk avec la meilleure similarité cosinus avec une image du dataset
                    descriptors.append(combined_chunks)

        self.paragraphsDescriptors = descriptors
        return descriptors