from PIL import Image
from transformers import pipeline

img = Image.open("https://telegra.ph/file/360a776a052a2ae26b28e.jpg")
classifier = pipeline("image-classification", model="Falconsai/nsfw_image_detection")
print(classifier(img))
