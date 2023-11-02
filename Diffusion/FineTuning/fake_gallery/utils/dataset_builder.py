import os
import requests
import json
from bs4 import BeautifulSoup
import os
import csv
import concurrent.futures
import argparse

def get_CAD_image_data(show_path):
    try:
        img_data = []
        show_response = requests.get(show_path)
        show_response.raise_for_status()
        show_html = show_response.text
        show_soup = BeautifulSoup(show_html, "html.parser")
        show_data_dict = json.loads(show_soup.find('script', {'id': '__NEXT_DATA__'}).string)
        img_list = show_data_dict['props']['pageProps']['projectObject']['images']
        artist = show_data_dict['props']['pageProps']['projectObject']['caption_artist'][0]['title']
        for img in img_list:
            img_data.append({"image":img['large'],"text":artist.replace(" ", "-")})
        return img_data
    except Exception as e:
        print(f"Failed to download {show_path}: {e}")

def get_TZVET_image_data(show_path):
    try:
        img_data = []
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
            img_data.append({"image":img_url, "text":artist})
        img_data
        return img_data        
    except Exception as e:
        print(f"Failed to download {show_path}: {e}")
    
def get_personal_image_data(show_path):
    try:
        img_data = []
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
                img_data.append({'image':img_url, 'text':artist})
        return img_data

    except Exception as e:
        print(f"Failed to download {show_path}: {e}")


def download_image(image_data, dataset_name, idx):
    # TODO - Build in our own pytorch classifier network to label
    # if images are - artwork, detail, or install
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

def download_images(img_data, dataset_name):
    os.makedirs(f"./{dataset_name}", exist_ok=True)
    metadata = [("file_name", "text")]
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(download_image, image, dataset_name, idx) for idx, image in enumerate(img_data)]
    
    for future in concurrent.futures.as_completed(futures):
        result = future.result()
        if result:
            metadata.append(result)
    
    with open(f"./{dataset_name}/metadata.csv", "w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(metadata)

def parse_arguments():
    description = "***img_grabber.py*** This script is designed to take a file of urls to gallery exhibitions from specific websites and turn it into an ImageFolder type dataset to be used for Stable Diffusion finetuning.\nCurrently supports these websites: - https://www.contemporaryartlibrary.org/ - https://tzvetnik.online/"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--input_file", "-i", help="Path to the input file. Must be a .txt file. The filename will be your dataset name as well. This file contains the full URLs to the shows desired.")
    return parser.parse_args()

def build_dataset(input_file):
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

def main():
    args = parse_arguments()
    input_file = args.input_file
    build_dataset(input_file)

if __name__ == "__main__":
    main()
