import os
import requests
import zipfile

# Correct EMNIST dataset URL
emnist_url = "https://biometrics.nist.gov/cs_links/EMNIST/matlab.zip"
download_dir = os.path.expanduser("~/.emnist")
zip_path = os.path.join(download_dir, "matlab.zip")

# Ensure download directory exists
os.makedirs(download_dir, exist_ok=True)

# Function to download the file with proper chunking
def download_file(url, dest_path):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(dest_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

# Check if file exists
if not os.path.exists(zip_path):
    print("Downloading EMNIST dataset...")
    try:
        download_file(emnist_url, zip_path)
        print("Download complete.")
    except Exception as e:
        print(f"Download failed: {e}")
        exit(1)
else:
    print("Dataset already downloaded.")

# Check and extract ZIP
try:
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        print("Extracting dataset...")
        zip_ref.extractall(download_dir)
        print("Extraction complete.")
except zipfile.BadZipFile:
    print("The downloaded file is not a valid ZIP. Deleting corrupted file.")
    os.remove(zip_path)
    print("Please re-run the script to download again.")
