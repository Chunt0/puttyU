import os

train_path = "./train/"
test_path = "./test/"

train_dir = [train_path + f + "/" for f in os.listdir(train_path) if os.path.join(train_path,f)]
test_dir = [test_path + f + "/" for f in os.listdir(test_path) if os.path.join(train_path,f)]

def rename_files(path):
    for d in path:
        for file in os.listdir(d):
            if file.endswith(".jpg"):
                src = d+file
                dst = d+"c"+file
                os.rename(src,dst)

rename_files(train_dir)
rename_files(test_dir)
