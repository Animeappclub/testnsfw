from PIL import Image
from transformers import pipeline

img = Image.open("image (1).png")
classifier = pipeline("image-classification", model="Falconsai/nsfw_image_detection")
print(classifier(img))
