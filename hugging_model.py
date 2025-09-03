from transformers import BlipProcessor, BlipForConditionalGeneration

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
import os
import cv2
def caption_generator(img):
    input=processor(img,return_tensors="pt")
    output=model.generate(**input)
    return processor.decode(output[0],skip_special_tokens=True)

