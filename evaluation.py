import re
import time
import os
from PIL import Image
from dotenv import load_dotenv
from google import genai
from logger import save_logs


load_dotenv("key.env")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
class Evaluation:
    def __init__(self, api_key=GEMINI_API_KEY):
        save_logs("Initializing Gemini Evaluation Model...\n")
        self.client = genai.Client(api_key=api_key)
        self.model_id = "gemini-3.1-flash-lite-preview"

    def evaluate_batch(self, paragraphs, images):
        notes = []
        for i in range(len(images)):
            image_path = images[i]
            paragraph = paragraphs[i]
            print(f"Evaluating paragraph: {paragraph}\nWith image: {image_path}\n")
            try:
                
                img = Image.open(image_path)
                
                prompt = f"""Compare this image with the following paragraph:
Paragraph: {paragraph}

Instructions:
1. Rate the relevance from 1 to 5 : 1 means not relevant, 5 means perfectly relevant.
2. Output ONLY the number. No text, no explanation.

Relevance score (1-5):"""
                print(f"Prompt for evaluation: {prompt}\n")
                response = self.client.models.generate_content(
                    model=self.model_id,
                    contents=[prompt, img]
                )
                print(response.text)
                
                output_text = response.text.strip()
                print(f"Model output: {output_text}")
                match = re.search(r'[1-5]', output_text)
                if match:
                    note = int(match.group())
                    notes.append(note)
                else:
                    save_logs(f"No number found in output: {output_text}\n")
                    notes.append(None)

                # on attend 4 sec pour éviter de faire trop de requêtes trop rapidement à l'API
                time.sleep(4)

            except Exception as e:
                print(f"Error on {image_path}: {e}\n")
                save_logs(f"Error on {image_path}: {e}\n")
                notes.append(None)
        return notes


if __name__ == "__main__":

    evaluator = Evaluation()
    notes = evaluator.evaluate_batch(["A landscape with mountains"], ["C:/Users/alex/Desktop/academique/cours polymtl/inf8801A/projet/data/laion-art-dl/image_4839.jpg"])
    print(notes)