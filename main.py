
# Load model directly
import torch
from PIL import Image
from transformers import AutoModelForImageClassification, ViTImageProcessor

img = Image.open("image (1).png")
model = AutoModelForImageClassification.from_pretrained("CompVis/stable-diffusion-safety-checker")
processor = ViTImageProcessor.from_pretrained("CompVis/stable-diffusion-safety-checker")
with torch.no_grad():
    inputs = processor(images=img, return_tensors="pt")
    outputs = model(**inputs)
    print(outputs)
    logits = outputs.logits
    print(logits)

predicted_label = logits.argmax(-1).item()
print(predicted_label)
output = model.config.id2label[predicted_label]
print(output)
