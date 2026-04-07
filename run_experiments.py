import os
from documentIllustrator import DocumentIllustrator
from evaluation import Evaluation
from logger import save_logs


input_documents = os.listdir(os.path.join(os.getcwd(), "textes"))
output_documents = os.listdir(os.path.join(os.getcwd(), "results"))
strategies = ["llm", "sliding_window", "best_matching_chunk", "truncate", "keywords"]

if __name__ == "__main__":
    illustrator = DocumentIllustrator()
    evaluation = Evaluation()
    for strategy in strategies:
        for filename in input_documents:
            title = os.path.splitext(filename)[0]
            outputfile = title + "_" + strategy + "_illustrated.md"
            if outputfile in output_documents:
                save_logs(f"{outputfile} already exists, skipping...\n")
                continue

            paragraphs, matching_images = illustrator.process(filename, strategy=strategy)
            img_paths = [matching_images[i][0]['path'] for i in range(len(paragraphs))]
            notes = evaluation.evaluate_batch(paragraphs, img_paths)
            save_logs(f"Evaluation notes for {filename} with strategy {strategy}: {notes}\n\n", 'notes.log')
