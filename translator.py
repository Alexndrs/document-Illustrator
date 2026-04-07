from transformers import MarianMTModel, MarianTokenizer
from langdetect import detect, DetectorFactory
import torch

class Translator:
    def __init__(self, target_lang="en", source_lang="fr"):
        model_name = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def translate_if_needed(self, texts: list[str]) -> list[str]:
        # on suppose que tous les paragraphes sont dans la même langue, on détecte la langue du premier texte
        try:
            lang = detect(texts[0])
            if lang == 'en':
                return texts
        except:
            # En cas d'échec (texte trop court par exemple) on suppose que c'est de l'anglais
            return texts
            
        # Si ce n'est pas de l'anglais, on traduit
        encoded = self.tokenizer(texts, return_tensors="pt",padding=True, truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            translated_tokens = self.model.generate(**encoded)
        translated_texts = self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
        return translated_texts