from datasetDescriptor import DatasetDescriptor
from inputDescriptor import InputDescriptor
from retrieval import Retrieval
from postprocessor import Postprocessor

from transformers import CLIPProcessor, CLIPModel
import torch
import os


if __name__ == "__main__":


    # modèle utilisé pour projeter dans l'espace latent
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)



    # construction du dataset descriptor

    datasetPath = os.path.join(os.getcwd(), "data")
    print(f"Construction du dataset issue du dossier: {datasetPath}")
    datasetDescriptor = DatasetDescriptor(datasetPath)
    datasetDescriptor.getImagesPaths()
    datasetDescriptor.projectDatasetWithClip(model, processor, batch_size=64, device=device, save_path=os.path.join(os.getcwd(),"projections.pt")) # fait en offline on récupère juste le checkpoint




    # construction de l'input descriptor

    inputPath = os.path.join(os.getcwd(), "textes/petite-sirene.pdf")
    inputDescriptor = InputDescriptor(inputPath)
    paragraphs = inputDescriptor.extractParagraphs()
    for i in range(len(paragraphs)):
        print(f"Paragraph {i} : {paragraphs[i]}")
        print('-----------------------------')
    inputDescriptor.extractDescriptors(inputDescriptor.paragraphs, processor, model, device)


    # retrieval
    retrieval = Retrieval(inputDescriptor, datasetDescriptor, top_k=5)
    matching_images = retrieval.match()

    # postprocessing
    postprocessor = Postprocessor(retrieval, inputDescriptor)
    postprocessor.rebuild()