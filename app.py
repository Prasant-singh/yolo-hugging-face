# Create a Flask web application that takes an image as input, performs object detection, generates a caption, and displays the annotated image and caption to the user.
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import os
from yolo_model import model_predict
import cv2
import hugging_model
from PIL import Image

# Initialize the Flask application
app = Flask(__name__)


# Configure the directories
upload_folder=os.path.join('static','uploads')
processed_folder=os.path.join('static','processed')
app.config["UPLOAD_FOLDER"] = upload_folder
app.config["PROCESSED_FOLDER"] = processed_folder
os.makedirs(upload_folder, exist_ok=True)
os.makedirs(processed_folder, exist_ok=True)


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        file = request.files['file']
        
        if file:
            filename = secure_filename(file.filename)
            save_path=os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(save_path)

            print(f"Image is stored at {save_path}")

            img=cv2.imread(save_path)
            image=Image.open(save_path)                                        # Using PIL to load the image for transformer
            caption=hugging_model.caption_generator(image)
            save_dir=app.config["PROCESSED_FOLDER"]
            processed_filepath=model_predict(img,filename,save_dir)

            return render_template('result.html', 
                               caption=caption.capitalize(), 
                               image_path=processed_filepath)
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)



