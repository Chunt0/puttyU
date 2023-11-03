import os
import requests
import json
from bs4 import BeautifulSoup
import os
import csv
import concurrent.futures
import argparse
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

def get_CAD_image_data(show_path):
    """
    Fetch and process image data from the Contemporary Art Daily website.

    Args:
        show_path (str): The URL to a gallery show on the CAD website.

    Returns:
        list: A list of dictionaries containing image URLs and associated text labels.
    """
    try:
        image_data_list = []
        show_response = requests.get(show_path)
        show_response.raise_for_status()
        show_html = show_response.text
        show_soup = BeautifulSoup(show_html, "html.parser")
        show_data_dict = json.loads(show_soup.find('script', {'id': '__NEXT_DATA__'}).string)
        img_list = show_data_dict['props']['pageProps']['projectObject']['images']
        artist = show_data_dict['props']['pageProps']['projectObject']['caption_artist'][0]['title']
        for img in img_list:
            image_data = add_CLIP_labels({"image":img['large'],"text":artist.replace(" ", "-")})
            image_data_list.append(image_data)
        return image_data_list
    except Exception as e:
        print(f"Failed to download {show_path}: {e}")

def get_TZVET_image_data(show_path):
    """
    Fetch and process image data from the Tzvetnik website.

    Args:
        show_path (str): The URL to a gallery show on the Tzvetnik website.

    Returns:
        list: A list of dictionaries containing image URLs and associated text labels.
    """
    try:
        image_data_list = []
        response = requests.get(show_path)
        response.raise_for_status()
        show_html = response.text
        show_soup = BeautifulSoup(show_html, "html.parser")
        div_tag = show_soup.find("div", class_="article__tags article--show")
        a_tags = div_tag.find_all('a')
        for a_tag in a_tags:
            if "artist" in a_tag.get("href"):
                artist = a_tag.text.replace(" ", "-")
                break
        action_texts = show_soup.find_all("action-text-attachment")
        for action_text in action_texts:
            img_url = action_text.get("url")
            image_data = add_CLIP_labels({"image":img_url, "text":artist})

            image_data_list.append(image_data)
        return image_data_list    
    except Exception as e:
        print(f"Failed to download {show_path}: {e}")
    
def get_personal_image_data(show_path):
    """
    Fetch and process personal image data from a given URL.

    Args:
        show_path (str): The URL to a personal gallery show.

    Returns:
        list: A list of dictionaries containing image file URLs and associated artist names.
    """
    try:
        image_data_list = []
        artist = show_path.split('/')[-2]
        response = requests.get(show_path)
        response.raise_for_status()
        show_html = response.text
        show_soup = BeautifulSoup(show_html, "html.parser")
        a_tags = show_soup.find_all('a')
        for a_tag in a_tags:
            tag_href = a_tag.get('href')
            if ".jpg" in tag_href:
                img_file = a_tag.text
                img_url = show_path+img_file
                image_data_list.append({'image':img_url, 'text':artist})
        
        return image_data_list

    except Exception as e:
        print(f"Failed to download {show_path}: {e}")

def add_CLIP_labels(image_data):
    """
    Add labels to an image using the CLIP model for image classification.

    Args:
        image_data (dict): A dictionary with keys 'image' for the image URL, and 'text' for the associated text.

    Returns:
        dict: The updated dictionary with additional labels from the CLIP model.
    """
    img_url = image_data['image']
    text = [image_data['text']]
    # Check if CUDA (GPU support) is available and set the device accordingly
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
  
    # Initialize the model and processor
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    try:
        response = requests.get(img_url, stream=True)
        response.raise_for_status()
        image = Image.open(response.raw).convert("RGB")

        labels = ["gallery", "artwork", "room", "photograph", "painting", 
                  "sculpture", "detail", "close-up", "drawing"]
        
        # Process inputs and move them to the device
        inputs = processor(text=labels, images=image, return_tensors="pt", padding=True).to(device)
        
        # Perform inference
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        
        # Move logits to CPU for further operations if necessary
        probs = logits_per_image.softmax(dim=1).squeeze().detach().cpu().numpy()
        probs = dict(zip(labels, probs))
        
        # Get top 2 labels
        top_2 = sorted(probs, key=probs.get, reverse=True)[:2]
        
        # Update the text with the top 2 labels
        text += top_2
        text = ", ".join(text)
        image_data['text'] = text
        
        return image_data
    except Exception as e:
        print(f"An error occurred: {e}")
        return image_data

def download_image(image_data, dataset_name, idx):
    """
    Download an image and save it to a specified dataset directory.

    Args:
        image_data (dict): A dictionary containing 'image' and 'text' keys.
        dataset_name (str): The name of the dataset directory to save the image.
        idx (int): The index of the image, used to generate the filename.

    Returns:
        tuple: A tuple containing the filename and text label of the downloaded image.
    """
    image_url = image_data['image']
    text = image_data['text']
    try:
        response = requests.get(image_url, stream=True)
        if response.status_code == 200:
            filename = f"{idx}.jpg"
            with open(f"./{dataset_name}/{filename}", 'wb') as file:
                for chunk in response.iter_content(8192):
                    file.write(chunk)
            return (filename, text)
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    return None

def download_images(image_data_list, dataset_name):
    """
    Download multiple images and save their metadata to a CSV file.

    Args:
        image_data_list (list): A list of image data dictionaries to be downloaded.
        dataset_name (str): The name of the dataset directory to save images and metadata.

    """
    os.makedirs(f"./{dataset_name}", exist_ok=True)
    metadata = [("file_name", "text")]
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(download_image, image, dataset_name, idx) for idx, image in enumerate(image_data_list)]
    
    for future in concurrent.futures.as_completed(futures):
        result = future.result()
        if result:
            metadata.append(result)
    
    with open(f"./{dataset_name}/metadata.csv", "w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(metadata)

def build_dataset(input_file):
    """
    Build an image dataset from a text file containing URLs to gallery shows.

    Args:
        input_file (str): The file path to a text file containing URLs.

    """
    dataset_name = input_file.strip(".txt")
    img_data = [] 
    with open(input_file) as f:
        for line in f.readlines():
            img_path = line.strip()
            if "contemporaryartlibrary" in line:
                show_data = get_CAD_image_data(img_path)
                img_data += show_data
            elif "tzvetnik" in line:
                show_data = get_TZVET_image_data(img_path)
                img_data += show_data
            elif "web.engr.oregonstate" in line:
                show_data = get_personal_image_data(img_path)
                img_data += show_data
            else:
                print(f"This url is not supported: {line.strip()}")
    if img_data:
        download_images(img_data, dataset_name)

def parse_arguments():
    """
    Parse command line arguments for the image grabber script.

    Returns:
        Namespace: An argparse Namespace with command line arguments.
    """
    description = "***img_grabber.py*** This script is designed to take a file of urls to gallery exhibitions from specific websites and turn it into an ImageFolder type dataset to be used for Stable Diffusion finetuning.\nCurrently supports these websites: - https://www.contemporaryartlibrary.org/ - https://tzvetnik.online/"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--input_file", "-i", help="Path to the input file. Must be a .txt file. The filename will be your dataset name as well. This file contains the full URLs to the shows desired.")
    return parser.parse_args()

def main():
    args = parse_arguments()
    input_file = args.input_file
    build_dataset(input_file)

if __name__ == "__main__":
    main()
