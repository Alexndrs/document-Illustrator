from datasetDescriptor import DatasetDescriptor
from inputDescriptor import InputDescriptor
from retrieval import Retrieval
from postprocessor import Postprocessor
from translator import Translator
from logger import save_logs

from transformers import CLIPProcessor, CLIPModel
import torch
import os


LOCAL_DATASET_PATH = os.path.join(os.getcwd(), "data")
HF_DATASET_NAMES = ["fantasyfish/laion-art"]



class DocumentIllustrator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)

        self.datasetDescriptor = DatasetDescriptor(LOCAL_DATASET_PATH, hf_dataset_names=HF_DATASET_NAMES, processor=self.processor)
        self.datasetDescriptor.getImagesPaths()
        self.datasetDescriptor.projectDatasetWithClip(self.model, batch_size=64, device=self.device, save_path=os.path.join(os.getcwd(),"projections.pt")) # fait en offline on récupère juste le checkpoint

    def process(self, filename, strategy):
        title = os.path.splitext(filename)[0]
        inputPath = os.path.join(os.getcwd(), "textes/" + filename)
        outputPath = os.path.join(os.getcwd(), "results/" + title + "_" + strategy + "_illustrated.md")



        save_logs(f"[Processing {filename} with strategy {strategy} on device {self.device}]\n")
        inputDescriptor = InputDescriptor(inputPath)
        paragraphs = inputDescriptor.extractParagraphs()
        translator = Translator() # on assume que le texte est en anglais ou en français auquel cas on le traduit en anglais.
        english_paragraphs = translator.translate_if_needed(paragraphs)
        inputDescriptor.extractDescriptors(english_paragraphs, self.processor, self.model, self.device, strategy=strategy)


        retrieval = Retrieval(inputDescriptor, self.datasetDescriptor, top_k=5)

        matching_images = retrieval.match(device=self.device)
        for i in range(len(paragraphs)):
            save_logs(f"Paragraph {i+1}:\nOriginal: {paragraphs[i]}\nTranslated: {english_paragraphs[i]}\nMatching images: {matching_images[i]}\n\n")


        postprocessor = Postprocessor(retrieval, inputDescriptor, output_path=outputPath)
        postprocessor.rebuild()
        save_logs(f"Results saved to {outputPath}\n\n\n\n")
        save_logs("_"*50)
        return (paragraphs, matching_images)

if __name__ == "__main__":
    # Exemple d'utilisation
    illustrator = DocumentIllustrator()
    paragraphs, matching_images = illustrator.process("test.txt", strategy="llm")
    for i in range(len(paragraphs)):
        paragraph, img_path = paragraphs[i], matching_images[i][0]['path']
        print(f"Paragraph: {paragraph}\nMatching image: {img_path}\n")