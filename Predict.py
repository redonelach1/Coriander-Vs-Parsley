from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
from PIL import Image, ImageDraw, ImageFont
import numpy as np

model = load_model("coriander_vs_parsley.h5")

def preproccess_image(image_path, target_size = (150,150)):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = img_array/255.0

    img_array = np.expand_dims(img_array, axis=0)
    return img_array

image_paths = ["PredictTest/coriander1.png",
               "PredictTest/coriander2.jpg",
               "PredictTest/coriander3.png",
               "PredictTest/parsley1.jpg",
               "PredictTest/parsley2.png",
               "PredictTest/parsley3.png",
               "PredictTest/parsley4.png"]
class_labels = ["Coriander","Parsley"]

for img_path in image_paths:
    img_array = preproccess_image(img_path)

    prediction = model.predict(img_array)
    predicted_class = class_labels[round(prediction[0][0])]
    confidence = prediction[0][0] if predicted_class == "Parsley" else 1-prediction[0][0]
    print(f"predicted class : {predicted_class} with confidence : {100*confidence:.2f}% for image path {img_path}")
    img = Image.open(img_path)
    center = img.size[0] // 2
    draw = ImageDraw.Draw(img)
    draw.text(xy=(center-30,160),text=f"{predicted_class}",fill=(0,0,0), font=ImageFont.truetype("arial.ttf", 50))
    img.show()