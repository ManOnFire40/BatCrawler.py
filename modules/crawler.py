
import http.client
import os
import re
import sys
import datetime
import time
from urllib.parse import urlencode, urljoin
import urllib.request
from urllib.error import HTTPError, URLError
from requests import Request
import tensorflow as tf
import numpy as np
from bs4 import BeautifulSoup
import urllib.request
from urllib.error import HTTPError, URLError
import re
from io import BytesIO
from PIL import Image
#################################################################### modifications
################################################################### OCR model loading
from Captcha_bypass_algorithm.OCR_model import load_model, load_prediction_model, decode_batch_predictions
import tensorflow as tf
import numpy as np
import pickle

with open('OCR_model.pkl', 'rb') as f:
 loaded_model = pickle.load(f)

prediction_model=None

if(loaded_model):
   prediction_model=loaded_model
else:
    # Load the OCR model
    ocr_model = load_model()
    # Load the prediction model for inference
    prediction_model = load_prediction_model()

       
# Example function to predict text from a batch of images
def predict_text_from_images(images):
    
    preds = prediction_model.predict(images)
    pred_texts = decode_batch_predictions(preds)
    return pred_texts




# Use the function
# Assuming you have preprocessed images ready
# images = [your preprocessed images batch here]
# predicted_texts = predict_text_from_images(images)
# print(predicted_texts)



prediction_model = load_prediction_model()
#prediction_model.load_weights('./Captcha_bypass_algorithm/ocr_model_weights.h5')
model_weights_path = 'D:\\Viruse project do not open\\SEMSTER 7\\Implementation\\TorCrawl.py-master\\modules\\Captcha_bypass_algorithm\\ocr_model_weights.weights.h5'


# Load the saved model


# Make predictions using the Joblib-loaded model
#y_pred_joblib = loaded_model_joblib.predict(X_test)

print("#####################################")
print(f"Loading model weights from: {model_weights_path}")



################################################################### image processing
# Constants for image dimensions
img_height = 50
img_width = 200

def preprocess_image(img):
    # 1. Decode the image to grayscale
    img = tf.io.decode_png(img, channels=1)
    # 2. Resize the image
    img = tf.image.resize(img, [img_height, img_width])
    # 3. Normalize pixel values to [0, 1] range
    img = tf.image.convert_image_dtype(img, tf.float32)
    # 4. Transpose the image (time dimension corresponds to width)
    img = tf.transpose(img, perm=[1, 0, 2])
    return img

# Preprocess multiple images for batch prediction
#def preprocess_images(image_paths):
#    return np.array([preprocess_image(path) for path in image_paths])


#################################################################### modifications



class Crawler:
    def __init__(self, website, c_depth, c_pause, out_path, logs, verbose):
        self.website = website
        self.c_depth = c_depth
        self.c_pause = c_pause
        self.out_path = out_path
        self.logs = logs
        self.verbose = verbose
        
        
        

    def excludes(self, link):
        """ Excludes links that are not required.

        :param link: String
        :return: Boolean
        """
        now = datetime.datetime.now().strftime("%Y%m%d")

        # BUG: For NoneType Exceptions, got to find a solution here
        if link is None:
            return True
        # Links
        elif '#' in link:
            return True
        # External links
        elif link.startswith('http') and not link.startswith(self.website):
            file_path = self.out_path + '/' + now + '_extlinks.txt'
            with open(file_path, 'w+', encoding='UTF-8') as lst_file:
                lst_file.write(str(link) + '\n')
            return True
        # Telephone Number
        elif link.startswith('tel:'):
            file_path = self.out_path + '/' + now + '_telephones.txt'
            with open(file_path, 'w+', encoding='UTF-8') as lst_file:
                lst_file.write(str(link) + '\n')
            return True
        # Mails
        elif link.startswith('mailto:'):
            file_path = self.out_path + '/' + now + '_mails.txt'
            with open(file_path, 'w+', encoding='UTF-8') as lst_file:
                lst_file.write(str(link) + '\n')
            return True
        # Type of files
        elif re.search('^.*\\.(pdf|jpg|jpeg|png|gif|doc)$', link,
                       re.IGNORECASE):
            return True

    def canonical(self, link):
        """ Canonicalization of the link.

        :param link: String
        :return: String 'final_link': parsed canonical url.
        """
        # Already formatted
        if link.startswith(self.website):
            return link
        # For relative paths with / in front
        elif link.startswith('/'):
            if self.website[-1] == '/':
                final_link = self.website[:-1] + link
            else:
                final_link = self.website + link
            return final_link
        # For relative paths without /
        elif re.search('^.*\\.(html|htm|aspx|php|doc|css|js|less)$', link,
                       re.IGNORECASE):
            # Pass to
            if self.website[-1] == '/':
                final_link = self.website + link
            else:
                final_link = self.website + "/" + link
            return final_link



        
    ################################################################# modifications
    
    
    def detect_captcha(self, content):
        """ Checks for CAPTCHA-related keywords in the page content. """
        if any(captcha in content.lower() for captcha in ['captcha', 'g-recaptcha', 'h-captcha']):
            return True
        else:
            return False
            
    
    

    def get_captcha_image(self):
        """
        Extracts and returns the CAPTCHA image from the crawler's current website if one is found.
        
        :return: CAPTCHA image as a PIL Image object or None if no CAPTCHA image is found.
        """
        try:
            # Fetch HTML content of the current website
            html_page = urllib.request.urlopen(self.website)
            soup = BeautifulSoup(html_page, 'html.parser')
            
            # Look for possible CAPTCHA images in the HTML content
            for img in soup.find_all('img'):
                img_src = img.get('src')
                img_alt = img.get('alt', '').lower()

                # Check if the image is likely a CAPTCHA (based on keywords in src or alt attributes)
                if img_src and ('captcha' in img_src.lower() or 'captcha' in img_alt):
                    # Complete URL if it's a relative path
                    if not img_src.startswith('http'):
                        img_src = urllib.parse.urljoin(self.website, img_src)
                    
                    # Download the image
                    response = urllib.request.urlopen(img_src)
                    img_data = response.read()
                    
                    # Convert to a PIL image for further processing
                    captcha_image = Image.open(BytesIO(img_data))
                    
                    print(f"CAPTCHA image found and loaded from {img_src}")
                    return captcha_image  # Returning as a PIL Image object

            print("No CAPTCHA image found on this page.")
            return None

        except (HTTPError, URLError) as e:
            print(f"Failed to load page or CAPTCHA image: {e}")
            return None
        
        
        
    
    def submit_captcha_solution(self, captcha_solution):
        """
        Submit the CAPTCHA solution to the website and continue crawling.
        
        :param captcha_solution: The decoded CAPTCHA text.
        :return: True if submission is successful, False otherwise.
        """
        try:
            # Load the page and find the CAPTCHA form
            html_page = urllib.request.urlopen(self.website)
            soup = BeautifulSoup(html_page, 'html.parser')
            
            
            # Find the CAPTCHA form
            captcha_form = soup.find('form')
            form_type = captcha_form.get_attribute("type")
            if not captcha_form or form_type=="hidden":
                print("CAPTCHA form not found.")
                return False
            
            form_action = captcha_form.get('action')
            if not form_action:
                print("No form action URL found.")
                return False

            # Prepare form data with the CAPTCHA solution
            form_data = {input_tag.get('name'): input_tag.get('value', '')
                         for input_tag in captcha_form.find_all('input')
                         if input_tag.get('name')}
            form_data['captcha'] = captcha_solution  # Fill in the CAPTCHA solution

            # Encode form data and submit it
            form_data_encoded = urlencode(form_data).encode('utf-8')
            action_url = urljoin(self.website, form_action)
            request = Request(action_url, data=form_data_encoded, method="POST")

            # Submit the form
            with urllib.request.urlopen(request) as response:
                if response.status == 200:
                    print("CAPTCHA solution submitted successfully.")
                    return True
                else:
                    print(f"CAPTCHA submission failed with status: {response.status}")
                    return False
        except Exception as e:
            print(f"Error submitting CAPTCHA solution: {e}")
            return False
    
    ################################################################# modifications

            
            
    def crawl(self):
        """ Core of the crawler.
        :return: List (ord_lst) - List of crawled links.
        """
        lst = set()
        ord_lst = []
        ord_lst.insert(0, self.website)
        ord_lst_ind = 0
        log_path = self.out_path + '/log.txt'

        if self.logs is True and os.access(log_path, os.W_OK) is False:
            print(f"## Unable to write to {self.out_path}/log.txt - Exiting")
            sys.exit(2)

        print(f"## Crawler started from {self.website} with "
              f"{str(self.c_depth)} depth crawl, and {str(self.c_pause)} "
              f"second(s) delay.")

        # Depth
        for index in range(0, int(self.c_depth)):

            # For every element of list.
            for item in ord_lst:
                html_page = http.client.HTTPResponse
                # Check if is the first element
                if ord_lst_ind > 0:
                    try:
                        #if "captcha" in response.text.lower() or "g-recaptcha" in response.text.lower():
                           # print("CAPTCHA detected!")

#######################################################################################################
                        content = html_page.read().decode('utf-8', errors='ignore')
                        is_captcha_exist = self.detect_captcha(content)
                        if is_captcha_exist:
                            img = self.get_captcha_image()
                            processed_img = preprocess_image(img)
                            captcha_solution = predict_text_from_images(processed_img)[0]
                                        # Submit the CAPTCHA solution
                            if self.submit_captcha_solution(captcha_solution):
                                print("Continuing with crawling after CAPTCHA submission.")
                               # Continue with the rest of the crawl
                            else:
                                print("Failed to submit CAPTCHA solution. Aborting.")
                                return []
#################################################################################################  
                        if item is not None:
                            html_page = urllib.request.urlopen(item)
                    except (HTTPError, URLError) as error:
                        print('## ERROR: Domain or link seems to be '
                              'unreachable. Add -v to see the verbose error.'
                              'Or write the full URL at -u argument!')
                        if self.verbose: print(error)
                        continue
                else:
                    try:
                        html_page = urllib.request.urlopen(self.website)
                        ord_lst_ind += 1
                    except (HTTPError, URLError) as error:
                        print('## ERROR: Domain or link seems to be '
                              'unreachable. Add -v to see the verbose error.'
                              'Or write the full URL at -u argument!')
                        if self.verbose: print(error)
                        ord_lst_ind += 1
                        continue

                try:
                    soup = BeautifulSoup(html_page, features="html.parser")
                except TypeError as err:
                    print(f"## Soup Error Encountered:: could to parse "
                          f"ord_list # {ord_lst_ind}::{ord_lst[ord_lst_ind]}")
                    continue

                # For each <a href=""> tag.
                for link in soup.findAll('a'):
                    link = link.get('href')

                    if self.excludes(link):
                        continue

                    ver_link = self.canonical(link)
                    if ver_link is not None:
                        lst.add(ver_link)

                # For each <area> tag.
                for link in soup.findAll('area'):
                    link = link.get('href')

                    if self.excludes(link):
                        continue

                    ver_link = self.canonical(link)
                    if ver_link is not None:
                        lst.add(ver_link)

                # TODO: For non-formal links, using RegEx
                # url_pattern = r'/(?:(?:https?|ftp|file):\/\/|www\.|ftp\.)(?:\([-A-Z0-9+&@#\/%=~_|$?!:,.]*\)|[-A-Z0-9+&@#\/%=~_|$?!:,.])*(?:\([-A-Z0-9+&@#\/%=~_|$?!:,.]*\)|[A-Z0-9+&@#\/%=~_|$])/igm'
                # html_content = urllib.request.urlopen(self.website).read().decode('utf-8')
                
                # if self.verbose:
                #     print("## Starting RegEx parsing of the page")
                # found_regex = re.findall(url_pattern, html_content)
                # for link in found_regex:
                #     if self.excludes(link):
                #         continue
                #     ver_link = self.canonical(link)
                #     if ver_link is not None:
                #         lst.add(ver_link)

                # TODO: For images
                # TODO: For scripts

                # Pass new on list and re-set it to delete duplicates.
                ord_lst = ord_lst + list(set(lst))
                ord_lst = list(set(ord_lst))

                if self.verbose:
                    sys.stdout.write("-- Results: " + str(len(ord_lst)) + "\r")
                    sys.stdout.flush()

                # Pause time.
                if (ord_lst.index(item) != len(ord_lst) - 1) and \
                        float(self.c_pause) > 0:
                    time.sleep(float(self.c_pause))

                # Keeps logs for every webpage visited.
                if self.logs:
                    it_code = html_page.getcode()
                    with open(log_path, 'w+', encoding='UTF-8') as log_file:
                        log_file.write(f"[{str(it_code)}] {str(item)} \n")

            print(f"## Step {str(index + 1)} completed "
                  f"with: {str(len(ord_lst))} result(s)")

        return ord_lst
    
