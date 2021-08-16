import os
from datetime import datetime

import numpy as np
from PIL import Image
from flask import Flask, request, flash, redirect, render_template
from flask_httpauth import HTTPBasicAuth
from werkzeug.utils import secure_filename

from tutorial.yolo_wrapper import YoloInferenceWrapper
from tutorial.utils.utils import allowed_file
from tutorial.settings import yolov3

app = Flask(__name__, static_url_path='/static', template_folder='templates')
auth = HTTPBasicAuth()

WORKING_DIR = os.path.dirname(__file__)
upload_folder = os.path.join(WORKING_DIR, 'static', 'img')
if not os.path.isdir(upload_folder):
    os.makedirs(upload_folder)
app.config['UPLOAD_FOLDER'] = upload_folder

detector = YoloInferenceWrapper(model_code=yolov3['weights'], cfg_code=yolov3['cfg'])


@app.route('/', methods=['GET', 'POST'])
# @auth.login_required
def predict():

    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['file']

        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file and allowed_file(file.filename):

            whole_process_time_begin = datetime.now()

            # save the image file in the upload_folder
            filename = secure_filename(file.filename)
            f_base = filename.split('.')[0]
            full_filename = os.path.join(upload_folder, filename)
            file.save(full_filename)

            label, label_proba, box = detector.predict(img_file=full_filename)
            image = Image.open(full_filename)

            image = image.crop((box[0], box[1], box[0]+box[2], box[1]+box[3]))

            fname = os.path.join(upload_folder, f"{f_base}_cropped.png")
            image.save(fname)

            whole_process_time_end = datetime.now()
            print(f": whole process time = {whole_process_time_end - whole_process_time_begin}")

            if label == 'abnormal':
                label = '異常'
            else:
                label = '正常'

            return render_template('results.html',
                                   label=label,
                                   label_proba=f"{label_proba}%",
                                   crop_image=f"{f_base}_cropped.png")

    return '''<form method="post", enctype=multipart/form-data> 
            Image: <input type=file name=file><br>
            <input type="submit" value="Upload">'''


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)
