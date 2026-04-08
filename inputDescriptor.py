'''
classe qui prend en entrée un fichier texte raw et qui s'occupe d'extraire les paragraphes et des descripteurs de ces paragraphes. 
'''
from PyPDF2 import PdfReader #lecture de pdf
import torch
import os
import time
import json
from google import genai
from dotenv import load_dotenv
from logger import save_logs


load_dotenv("key.env")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


class InputDescriptor:
    def __init__(self, inputPath : str):
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

    def extractDescriptors(self, paragraphs, processor, model, device, strategy="best_matching_chunk", api_key=GEMINI_API_KEY):
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

        if strategy == "llm" or strategy == "keywords":        
            client = genai.Client(api_key=api_key)
            model_id = "gemini-3.1-flash-lite-preview"
        

        descriptors = []
        processed_texts = []


        if strategy in ["llm", "keywords"]:
            # on regroupe les paragraphes pour traiter par batch et limiter les appels à l'API
            batch_size = 8

            for i in range(0, len(paragraphs), batch_size):
                batch = paragraphs[i : i + batch_size]
                
                if strategy == "llm":
                    prompt_content = f'''Return a JSON object with a key "results" containing a list of {len(batch)} strings.
Each string must be a CLIP-optimized caption (5-12 words, visible elements only, no abstraction) for the corresponding text.

Texts to process:
{json.dumps(batch, indent=2)}

Format: {{"results": ["caption 1", "caption 2", ...]}}'''
            
                else: # keywords
                    prompt_content = f'''Return a JSON object with a key "results" containing a list of {len(batch)} strings.
    Each string must be exactly 5 visual keywords separated by commas for the corresponding text.

    Texts to process:
    {json.dumps(batch, indent=2)}

    Format: {{"results": ["key1, key2, ...", "key1, key2, ...", ...]}}'''
                        

                try:
                    response = client.models.generate_content(
                        model=model_id,
                        contents=[prompt_content],
                        config={'response_mime_type': 'application/json'}
                    )
                    batch_results = json.loads(response.text).get("results", [])
                    while len(batch_results) < len(batch):
                        batch_results.append("error processing text")
                    
                    processed_texts.extend(batch_results[:len(batch)])
                    for original, transformed in zip(batch, batch_results):
                        save_logs(f"Strategy: {strategy}\nOriginal: {original[:100]}...\nResult: {transformed}\n")
                    
                    time.sleep(20)
                except Exception as e:
                    
                    # si erreur 429 on attend 60 secondes avant de réessayer
                    if "429" or "503" in str(e):
                        # 429 Too Many Requests or 503 Service Unavailable sont des erreurs de rate limit ou de surcharge du serveur
                        print("Rate limit exceeded. Waiting for 60 seconds before retrying...")
                        time.sleep(60)
                        try:
                            response = client.models.generate_content(
                                model=model_id,
                                contents=[prompt_content],
                                config={'response_mime_type': 'application/json'}
                            )
                            batch_results = json.loads(response.text).get("results", [])
                            while len(batch_results) < len(batch):
                                batch_results.append("error processing text")
                            
                            processed_texts.extend(batch_results[:len(batch)])
                            for original, transformed in zip(batch, batch_results):
                                save_logs(f"Strategy: {strategy}\nOriginal: {original[:100]}...\nResult: {transformed}\n")
                            
                            time.sleep(2)
                        except Exception as e:
                            print(f"Error in batch {i}: {e}")
                            processed_texts.extend([""] * len(batch))
                    else:
                        print(f"Error in batch {i}: {e}")
                        processed_texts.extend([""] * len(batch))

            for text_input in processed_texts:
                inputs = processor(text=[text_input], return_tensors="pt", padding=True, truncation=True, max_length=77).to(device)
                with torch.no_grad():
                    text_outputs = model.text_model(**inputs)
                    pooled = text_outputs.pooler_output
                    features = model.text_projection(pooled)
                    features /= features.norm(p=2, dim=-1, keepdim=True)
                    descriptors.append(features)

        elif strategy == "truncate":
            for p in paragraphs:
                inputs = processor(text=[p], return_tensors="pt", padding=True, truncation=True, max_length=77).to(device)
                with torch.no_grad():
                    text_outputs = model.text_model(**inputs)
                    features = model.text_projection(text_outputs.pooler_output)
                    features /= features.norm(p=2, dim=-1, keepdim=True)
                    descriptors.append(features)

        
        elif strategy in ["sliding_window", "best_matching_chunk"]:
            for p in paragraphs:
                token_ids = processor.tokenizer(p, truncation=False, return_tensors="pt")['input_ids'][0]
                chunk_size, overlap = 77, 10
                chunks = [token_ids[i:i+chunk_size] for i in range(0, len(token_ids), chunk_size - overlap)]
                
                chunk_features = []
                for chunk in chunks:
                    if len(chunk) < chunk_size:
                        chunk = torch.cat([chunk, torch.zeros(chunk_size - len(chunk), dtype=torch.long)])
                    input_ids = chunk.unsqueeze(0).to(device)
                    with torch.no_grad():
                        text_outputs = model.text_model(input_ids=input_ids)
                        feat = model.text_projection(text_outputs.pooler_output)
                        feat /= feat.norm(p=2, dim=-1, keepdim=True)
                        chunk_features.append(feat)
                
                combined_chunks = torch.cat(chunk_features, dim=0)
                if strategy == "sliding_window":
                    mean_feat = combined_chunks.mean(dim=0, keepdim=True)
                    mean_feat /= mean_feat.norm(p=2, dim=-1, keepdim=True)
                    descriptors.append(mean_feat)
                else:
                    descriptors.append(combined_chunks)

        self.paragraphsDescriptors = descriptors
        return descriptors