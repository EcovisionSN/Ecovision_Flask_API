from flask import Flask, request, jsonify
import cloudinary
import requests
import os
from cloudinary import uploader
from functions import *
 
cloudinary.config( 
  cloud_name = "dbg1vrj7e", 
  api_key = "275848465849392", 
  api_secret = "xKHW44eHfbWsSkuLFPpI9v_vmCA" 
)

dir = os.getcwd()

app = Flask(__name__)


@app.route('/analyse', methods=['POST'])
def analyse_prediction():
    image_url = request.form['url']
    image_path = os.path.join(dir, "image.png")
    try:
        image_data = requests.get(str(image_url))
        image_content = image_data.content
        
        with open(image_path, "wb") as f:
            f.write(image_content)

    except Exception as e:
        pass

    prediction_path = predict(image_path)
    figure_path, absorption_carbone_kg_an, trees_number = analyse(prediction_path)

    result = cloudinary.uploader.upload(prediction_path)
    prediction_url_cloudinary = result['url']

    result = cloudinary.uploader.upload(figure_path)
    figure_url_cloudinary = result['url']

    os.remove(image_path)
    os.remove(prediction_path)
    os.remove(figure_path)

    return jsonify({"base_image": image_url,
                    "prediction_url": prediction_url_cloudinary, 
                    "figure_url": figure_url_cloudinary, 
                    "absorption_carbone_kg_an": absorption_carbone_kg_an, 
                    "trees_number": trees_number})


@app.route('/upload/image', methods=['POST'])
def upload_file():
    f = request.files['fichier']
    result = cloudinary.uploader.upload(f)
    url = result['url']

    return jsonify({"url": url})


if __name__ == '__main__':
    app.run(port=8080, debug=True)
