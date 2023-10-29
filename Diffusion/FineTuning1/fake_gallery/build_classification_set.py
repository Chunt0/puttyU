# We are building a classifier neural network.
# This is going to require us to contstruct a data set with appropriate tags
# We need to grab a lot of images from the TZVET data set.
# We will have 3 class categories
# - installation
# - artwork
# - detail
#
# To try to make sure these tags exist in a more empty semantic space in the SD model
# we will spell them backwards. We are gonna have to do this on a lot of images....
import os
from PIL import Image
import matplotlib.pyplot as plt

TZVET_PATH = "./TZVET/TZVET_img/"
artwork_path = "./class_img/artwork/"
install_path = "./class_img/install/"
detail_path = "./class_img/detail/"

TZVET_dir = [TZVET_PATH+f+"/" for f in os.listdir(TZVET_PATH) if os.path.isdir(os.path.join(TZVET_PATH,f))]

for d in TZVET_dir[:5]:
    for filename in os.listdir(d):
        if filename.endswith(".jpg"):
            try:
                img = Image.open(d+filename)
                plt.imshow(img)
                plt.axis('off')  # Optional: Turn off axis labels and ticks
                plt.show()
            except Exception as e:
                print(f"Error: {str(e)}")


    
