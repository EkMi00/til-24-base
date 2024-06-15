from typing import List
from io import BytesIO
from PIL import Image
from transformers import pipeline

class VLMManager:
    def __init__(self):
        # initialize the model here
        pass

    def identify(self, image: bytes, caption: str) -> List[int]:
        # perform object detection with a vision-language model

        model_path = "local/owlv2-base-patch16-ensemble"
        detector = pipeline(model=model_path, task="zero-shot-object-detection") #model_path
        
        predictions = detector(
            Image.open(BytesIO(image)),
            candidate_labels=[caption],
            threshold=1e-2,
            top_k=1,
            device=0
        )

        if bool(predictions):
            box = predictions[0]['box']
            xmin, ymin, xmax, ymax = box.values()
            return [xmin, ymin, xmax - xmin, ymax - ymin] 
        else:
            return [0, 0, 0, 0]
