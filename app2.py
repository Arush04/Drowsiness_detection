import os
import csv
import json
from flask import Flask, render_template, request, send_from_directory
from eye_detection2 import detect_eyes

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/upload", methods=["POST"])
def upload_file():
    if request.method == "POST":
        # Check if the POST request has the file part
        if 'file' not in request.files:
            return "No file part"
        
        file = request.files['file']

        # If the user does not select a file, the browser may also
        # submit an empty part without a filename
        if file.filename == '':
            return "No selected file"

        # Save the uploaded file to the uploads folder
        file_path = os.path.join("uploads", file.filename)
        file.save(file_path)

        # Determine the type of video uploaded (face or car)
        video_type = request.form['video_type']

        if video_type == 'face':
            # Run face detection script
            output_csv = os.path.join("uploads", os.path.splitext(file.filename)[0] + "_face.csv")
            detect_eyes(file_path, output_csv)
        elif video_type == 'car':
            # Run car detection script
            command = "python detect.py --weights yolov5s.pt --source " + file_path + " --view-img --save-csv"
            os.system(command)
        else:
            return "Invalid video type"

        # Return detection completion message
        return render_template('result.html', video_filename=file.filename)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('uploads', filename)

if __name__ == "__main__":
    app.run(debug=True)
