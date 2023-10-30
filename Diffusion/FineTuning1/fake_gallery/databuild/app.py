import os
from flask import Flask, render_template, request, redirect, url_for
from PIL import Image
import csv
import shutil

app = Flask(__name__)

TZVET_PATH = "./static/TZVET/TZVET_img/"
TZVET_dir = [TZVET_PATH + f + "/" for f in os.listdir(TZVET_PATH) if os.path.isdir(os.path.join(TZVET_PATH, f))]
classes = ["install", "artwork", "detail"]
img_files = []
for d in TZVET_dir:
    for file in os.listdir(d):
        if file.endswith(".jpg"):
            img_files.append(d + file)

class_counts = [0] * len(classes)
labeled_imgs = []
idx = 0

@app.route('/')
def index():
    return render_template('index.html', img_file=img_files[idx], classes=classes)

@app.route('/submit', methods=['POST'])
def submit():
    global idx, class_counts, labeled_imgs, img_files

    img_filename = img_files[idx]
    try:
        user_input = int(request.form.get('user_input'))
    except:
        user_input = -1
    
    if 0 <= user_input < len(classes):
        label = classes[user_input]
        class_counts[user_input] += 1
        if class_counts[user_input] < 401:
            dst = f"./static/img/train/{label}/{idx}.jpg"
        elif class_counts[user_input] < 501:
            dst = f"./static/img/test/{label}/{idx}.jpg"
        else:
            print(f"All {label} files have been labeled")
            dst = None

        if dst:
            try:
                shutil.copy(img_filename, dst)
            except:
                pass
        labeled_imgs.append((img_filename, label))
    else:
        label = ""
        print(f"Incorrect label: {img_filename}")

    with open("./metadata.csv", "w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(labeled_imgs)
    
    idx += 1
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
