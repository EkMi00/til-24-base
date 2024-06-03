from typing import List
from transformers import pipeline

class VLMManager:
    def __init__(self):
        # initialize the model here
        pass

    def identify(self, image: bytes, caption: str) -> List[int]:
        # perform object detection with a vision-language model

        model_path = "local/custom_owl_vit_1"
        detector = pipeline(model=model_path, task="zero-shot-object-detection") #model_path

        predictions = detector(
            image,
            candidate_labels=[caption],
            threshold=0.10,
            top_k=3
        )
        
        xmin, ymin, xmax, ymax = max(predictions, key=lambda x: x['score'])

        return [xmin, ymin, xmax - xmin, ymax - ymin]
