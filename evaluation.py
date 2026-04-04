from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image
import torch
from transformers import BitsAndBytesConfig


model_name = "Qwen/Qwen2-VL-2B-Instruct"

prompt = """Ton rôle est d'évaluer la pertinence d'une illustration par rapport à un paragraphe donné.
Entrée : [IMAGE] + {paragraphe}
Instructions : 1. Analyse les éléments visuels clés de l'image (sujet, personnage, environnement...).
2. Compare-les avec les informations du paragraphe.
3. Donne une note de 1 à 5, où 1 signifie "pas du tout pertinent" et 5 signifie "très pertinent". 

Format de réponse : "note"

Tu ne dois retourner que la note, sans explication ni commentaire supplémentaire.
"""


def evaluate_batch(model_name, paragraphs, images, quantization = True) :
    print("loading model")
    if quantization :
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        model = Qwen2VLForConditionalGeneration.from_pretrained(model_name, quantization_config=quantization_config, device_map="auto")
    else :
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype="auto", device_map="auto")
    
    processor = AutoProcessor.from_pretrained(model_name)
    
    print("processing inferences")
    notes = []
    for (paragraph, image_path) in zip(paragraphs, images) :
        image = Image.open(image_path)  #vérifier le chemin d'accès
        messages = [{"role": "user","content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},],}]
        
        #inférence
        text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text_prompt], images=[image], padding=True, return_tensors="pt").to("cuda")
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        output_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
        note = float(output_text)
        notes.append(note)
    return notes
#calculer ensuite moyenne, variance etc, faire varier les datasets,techniques, documents


#tests

paragraphs = ["""Bien loin dans la mer, il est un endroit où l’eau est, pure comme le verre le plus 
transparent, mais si profonde qu’il serait inutile d’y jeter l’ancre. Il faudrait y entasser 
une quantité infinie de tours d’église les unes sur les autres pour mesurer la distance 
séparant la surface du fond. 
C’est là que demeure le peuple de la mer. Sur un fond de sable blanc des plantes et 
des arbres bizarres y croissent, si souple que le moindre mouvement de l’eau les fait 
onduler et bouger comme s’ils étaient vivants.
Tous les poissons, grands et petits, nagent entre les branches comme les oiseaux dans 
l’air."""]
images = ['data/paysages/1091675.jpg']

#evaluate_batch(model_name, paragraphs, images)