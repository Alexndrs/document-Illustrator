from datasetDescriptor import DatasetDescriptor
from inputDescriptor import InputDescriptor
from postprocessor import Postprocessor
from retrieval import Retrieval

from transformers import CLIPProcessor, CLIPModel
import torch
import os


if __name__ == "__main__":

    datasetPath = os.path.join(os.getcwd(), "data")
    print(f"Dataset path: {datasetPath}")

    datasetDescriptor = DatasetDescriptor(datasetPath)
    datasetDescriptor.getImagesPaths()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)

    datasetDescriptor.projectDatasetWithClip(model, processor, batch_size=64, device=device, save_path=os.path.join(os.getcwd(),"projections.pt"))