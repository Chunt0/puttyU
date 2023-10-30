import os
from flask import Flask, render_template, request, redirect, url_for
import shutil
import csv

app = Flask(__name__)


TZVET_PATH = "./static/TZVET/TZVET_img/"
TZVET_dir = [TZVET_PATH+f+"/" for f in os.listdir(TZVET_PATH) if os.path.isdir(os.path.join(TZVET_PATH,f))]
classes = ["install", "artwork", "detail"]
img_files = []
for d in TZVET_dir[:2]:
    for file in os.listdir(d):
        if file.endswith(".jpg"):
            img_files.append(d+file)
idx = 0
class_counts= [0,0,0]
labeled_imgs = [("file_name", "label")]

@app.route('/')
def index():
    img = img_files[idx]
    print(img)
    return render_template('index.html', img_file=img)

@app.route('/submit', methods=['POST'])
def submit():
    global idx, class_counts 
    img_filename = img_files[idx]
    user_input = int(request.form.get('user_input'))
    class_counts[user_input] += 1
    if user_input == 0:
        class_counts[0] += 1
        label = classes[user_input]
        if class_counts[0] < 401:
            dst = "./static/img/train/"+label+f"/{idx}.jpg"
            try:
                shutil.copy(img_filename,dst)
            except:
                pass
        elif class_counts[0] < 501:
            dst = "./static/img/test/"+label+f"/{idx}.jpg"
            try:
                shutil.copy(img_filename,dst)
            except:
                pass
        else:
            print(f"All {label} files have been labeled")
    elif user_input == 1:
        class_counts[1] += 1
        label = classes[user_input]
        if class_counts[1] < 401:
            dst = "./static/img/train/"+label+f"/{idx}.jpg"
            try:
                shutil.copy(img_filename,dst)
            except:
                pass
        elif class_counts[1] < 501:
            dst = "./static/img/test/"+label+f"/{idx}.jpg"
            try:
                shutil.copy(img_filename,dst)
            except:
                pass
        else:
            print(f"All {label} files have been labeled")
    elif user_input == 2:
        class_counts[2] += 1
        label = classes[user_input]
        if class_counts[2] < 401:
            dst = "./static/img/train/"+label+f"/{idx}.jpg"
            try:
                shutil.copy(img_filename,dst)
            except:
                pass
        elif class_counts[2] < 501:
            dst = "./static/img/test/"+label+f"/{idx}.jpg"
            try:
                shutil.copy(img_filename,dst)
            except:
                pass
        else:
            print(f"All {label} files have been labeled")
    else:
        label = ""
        print(f"incorrect label: {img_filename}")

    labeled_imgs.append((img_filename,label))

    with open("./metadata.csv", "w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(labeled_imgs)
    
    idx += 1
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
