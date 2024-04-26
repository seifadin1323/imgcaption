import os
from flask import Flask, jsonify, request
import cv2
import numpy as np
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
import torch
from PIL import Image

app = Flask(__name__)
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
@app.route('/api/image_caption', methods=['PUT'])
def image_caption():
    try:
        # Get image file from request
        file = request.files['image']

        # Open image and preprocess
        image = Image.open(file)
        if image.mode != "RGB":
            image = image.convert(mode="RGB")

        # Perform image captioning
        caption = predict_caption(image)
        print('cptionnnnn' + caption)
        # Return caption
        return jsonify(caption)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
# Function to predict image caption
def predict_caption(image):
    inputs = feature_extractor(images=image, return_tensors="pt")
    pixel_values = inputs.pixel_values.to(device)

    output_ids = model.generate(pixel_values, max_length=16, num_beams=4)

    captions = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    captions = [caption.strip() for caption in captions]
    return captions[0] if captions else "No caption generated"
@app.route('/')
def hello():
    return 'Hello, caption'
if __name__ == '__main__':
    app.run(debug=True, port=8000)
