import os
import requests
import json
from bs4 import BeautifulSoup
from huggingface_hub import Repository
import os
import csv
import concurrent.futures
import argparse

def get_CAD_image_data(show_url):
    try:
        img_data = []
        show_response = requests.get(show_url)
        show_response.raise_for_status()
        show_html = show_response.text
        show_soup = BeautifulSoup(show_html, "html.parser")
        show_data_dict = json.loads(show_soup.find('script', {'id': '__NEXT_DATA__'}).string)
        img_list = show_data_dict['props']['pageProps']['projectObject']['images']
        artist = show_data_dict['props']['pageProps']['projectObject']['caption_artist'][0]['title']
        for img in img_list:
            img_data.append({"image":img['large'],"label":artist.replace(" ", "-")})
        return img_data
    except Exception as e:
        print(f"Failed to download {show_url}: {e}")

def get_TZVET_image_data(show_url):
    try:
        img_data = []
        response = requests.get(show_url)
        response.raise_for_status()
        show_html = response.text
        soup = BeautifulSoup(show_html, "html.parser")
        
        div_tag = soup.find("div", class_="article__tags article--show")
        a_tags = div_tag.find_all('a')
        for a_tag in a_tags:
            if "artist" in a_tag.get("href"):
                artist = a_tag.text.replace(" ", "-")

        action_texts = soup.find_all("action-text-attachment")
        for action_text in action_texts:
            img_url = action_text.get("url")
            img_data.append({"image":img_url, "label":artist})
        img_data
        return img_data        
    except Exception as e:
        print(f"Failed to download {show_url}: {e}")

def download_image(image_data, dataset_name, idx):
    # TODO - Build in our own pytorch classifier network to label
    # if images are - artwork, detail, or install
    image_url = image_data['image']
    label = image_data['label']
    try:
        response = requests.get(image_url, stream=True)
        if response.status_code == 200:
            filename = f"{idx}.jpg"
            with open(f"./{dataset_name}/{filename}", 'wb') as file:
                for chunk in response.iter_content(8192):
                    file.write(chunk)
            return (filename, label)
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    return None

def download_images(img_data, dataset_name):
    os.makedirs(f"./{dataset_name}", exist_ok=True)
    metadata = [("file_name", "labels")]
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(download_image, image, dataset_name, idx) for idx, image in enumerate(img_data)]
    
    for future in concurrent.futures.as_completed(futures):
        result = future.result()
        if result:
            metadata.append(result)
    
    with open(f"./{dataset_name}/metadata.csv", "w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(metadata)

######################################################################
# TODO this needs testing
def upload_dataset(dataset_name, token):
    # Grab token
    with open(token, 'r') as f:
        f = f.readlines()
        organization = f[0].strip()
        token = f[1].strip()

    # Define local and remote repository paths
    dataset_path = './' + dataset_name + '/'
    local_repo_path = "./temp_dataset_repo"
    remote_repo_name = f"{organization}/{dataset_name}"

    # Initialize a new repository
    repo = Repository(local_repo_path, clone_from=remote_repo_name, use_auth_token=token)
    repo.git_pull()

    # Copy dataset files to repository
    for file_name in os.listdir(dataset_path):
        full_file_path = os.path.join(dataset_path, file_name)
        if os.path.isfile(full_file_path):
            repo.lfs_track(full_file_path)
            os.system(f"cp {full_file_path} {local_repo_path}/{file_name}")

    # Commit and push changes
    repo.git_add(auto_lfs_track=True)
    commit_message = "Upload dataset"
    repo.git_commit(commit_message)
    repo.git_push()

    print(f"Dataset successfully uploaded to: https://huggingface.co/{remote_repo_name}")
#########################################################################################

def parse_arguments():
    description = "***img_grabber.py*** This script is designed to take a file of urls to gallery exhibitions from specific websites and turn it into an ImageFolder type dataset to be used for Stable Diffusion finetuning.\nCurrently supports these websites: - https://www.contemporaryartlibrary.org/ - https://tzvetnik.online/"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--input_file", "-i", help="Path to the input file. Must be a .txt file. The filename will be your dataset name as well. This file contains the full URLs to the shows desired.")
    # This feature does not work yet
    #parser.add_argument("--token", help="Must provide path to write token file. This file must have the organization as the first line and the write token as the second. Will trigger the push_to_hub pathway.")
    return parser.parse_args()


def main():
    args = parse_arguments()
    input_file = args.input_file
    dataset_name = input_file.strip(".txt")
    token = args.token
    img_data = [] 
    with open(input_file) as f:
        for line in f.readlines():
            if "contemporaryartlibrary" in line:
                img_url = line.strip()
                show_data = get_CAD_image_data(img_url)
                img_data += show_data
            elif "tzvetnik" in line:
                img_url = line.strip()
                show_data = get_TZVET_image_data(img_url)
                img_data += show_data
            else:
                print(f"This url is not supported: {line.strip()}")
    if img_data:
        download_images(img_data, dataset_name)

    # This feature is not working
    #if token:
        #upload_dataset(dataset_name, token)

if __name__ == "__main__":
    main()
