
from PIL import ImageEnhance , Image
import torch
import pandas as pd
import tabulate
import cv2
import numpy as np
import torch







def preprocess_data(image) :

  if image.mode != "RGB":
        image = image.convert("RGB")
  # Enhance the contrast
  enhancer = ImageEnhance.Contrast(image)
  enhanced_image = enhancer.enhance(2.2)  # Adjust the enhancement factor as desired


  # Enhance the brightness
  enhancer = ImageEnhance.Brightness(enhanced_image)
  brightened_image = enhancer.enhance(1)
  image = np.array(brightened_image)

  # Apply denoising
  denoised_image = cv2.fastNlMeansDenoisingColored(image, None, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21)

  # Convert the denoised image back to PIL Image format
  denoised_image = Image.fromarray(denoised_image)
  return denoised_image
def inference_model(args) :
      model, processor, img  , i= args
      img_pr= preprocess_data(img)
      pixel_values = processor(img_pr, return_tensors="pt").pixel_values
                #tensor_image = transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to device

      generated_ids = model.generate(pixel_values, return_dict_in_generate=True, output_scores=True)
      generated_text = processor.batch_decode(generated_ids.sequences, skip_special_tokens=True)
      return generated_text[0] ,i




