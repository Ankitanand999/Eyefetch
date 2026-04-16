from flask import Flask, render_template,request, jsonify, send_file,Response
import requests
from ultralytics import YOLO
from bs4 import BeautifulSoup
import re
import os
from urllib.parse import urljoin, urlparse
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
import time
import zipfile
import pandas as pd

# Image Processing
import cv2
from io import BytesIO


app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
DOWNLOAD_FOLDER = "downloads"
os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)
MIN_IMAGE_SIZE_BYTES = 5 * 1024 # 5 KB



model = YOLO("yolov8n.pt")

# Global state management
camera = None
object_count = 0
system_active = False

def gen_frames():
    global camera, object_count, system_active
    
    # Initialize camera if not already running
    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(0)
    
    system_active = True

    while system_active:
        success, frame = camera.read()
        if not success:
            break
        
        results = model(frame)
        object_count = len(results[0].boxes)
        annotated_frame = results[0].plot()

        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    # Ensure camera is released when the loop breaks
    if camera:
        camera.release()
        camera = None

@app.route('/yolo')
def yolo():
    return render_template('yolo.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_feed', methods=['POST'])
def stop_feed():
    global system_active, camera, object_count
    system_active = False
    if camera:
        camera.release()
        camera = None
    object_count = 0
    return jsonify({"message": "System shutdown successful", "status": "OFFLINE"})

@app.route('/status')
def status():
    return jsonify({
        "status": "ACTIVE" if system_active else "OFFLINE",
        "count": object_count
    })


# ==============================================================================
# 1. HELPER FUNCTIONS (REUSED FROM PREVIOUS CONVERSATION)
# ==============================================================================

def scrape_images(url):
    """Scrapes image sources from a dynamically loaded page using Selenium."""
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.6778.265 Safari/537.36")
    
    # Use ChromeDriverManager to automatically handle the ChromeDriver executable
    try:
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
    except Exception as e:
        print(f"Error setting up WebDriver: {e}")
        return pd.DataFrame() # Return empty DataFrame on setup failure

    try:
        driver.get(url)
        # Wait until the body element is loaded (basic check)
        WebDriverWait(driver, 30).until(lambda d: d.find_element(By.TAG_NAME, "body"))
        
        # Scroll down a few times to load lazy-loaded images
        SCROLL_PAUSE_TIME = 1.5
        last_height = driver.execute_script("return document.body.scrollHeight")
        MAX_SCROLLS = 3
        scroll_count = 0
        while scroll_count < MAX_SCROLLS:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(SCROLL_PAUSE_TIME)
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height: 
                break
            last_height = new_height
            scroll_count += 1
            
        html = driver.page_source
        driver.quit()
    except Exception as e:
        print(f"Selenium scraping failed: {e}")
        try: driver.quit() 
        except: pass
        return pd.DataFrame()

    soup = BeautifulSoup(html, "html.parser")
    places = soup.find_all("img")
    image_data = []
    base_url_parts = urlparse(url)
    base_url = f"{base_url_parts.scheme}://{base_url_parts.netloc}"
    
    for img in places:
        alt_text = img.get("alt", "No alt text")
        src_link = img.get("src", "No src link")
        
        # Resolve relative URLs
        if src_link.startswith("//"): src_link = "https:" + src_link
        elif src_link.startswith("/"): src_link = urljoin(base_url, src_link)
        
        # Simple filter for data URIs and empty links
        if len(src_link) < 10 or src_link.startswith("data:"): continue
        
        image_data.append({"alt": alt_text, "src": src_link})
        
    df = pd.DataFrame(image_data).drop_duplicates(subset=['src'])
    return df

def download_images(df, folder_path):
    # This helper is kept for the original /get_images route, 
    # but is NOT used by the new /scrape_all route, which only fetches URLs.
    os.makedirs(folder_path, exist_ok=True)
    failed_downloads = []
    log_path = os.path.join(folder_path, "failed_downloads.log")
    if os.path.exists(log_path): os.remove(log_path)
    
    for index, row in df.iterrows():
        img_url = row['src']
        if not re.search(r'\.(jpg|jpeg|png|gif|webp|bmp)(\?.*)?$', img_url, re.I): continue
        try:
            head_response = requests.head(img_url, timeout=5, allow_redirects=True)
            head_response.raise_for_status()
            content_length = int(head_response.headers.get('content-length', 0))
            if content_length < MIN_IMAGE_SIZE_BYTES and content_length != 0: continue
            
            response = requests.get(img_url, timeout=10)
            response.raise_for_status()
            
            clean_alt = row['alt'][:50].replace('/', '_').replace(' ', '_').strip()
            extension = os.path.splitext(urlparse(img_url).path)[1]
            if not extension or len(extension) > 5: extension = '.jpg'
            file_name = f"{index}_{clean_alt}{extension}"
            file_path = os.path.join(folder_path, file_name)
            
            with open(file_path, "wb") as f: f.write(response.content)
        except Exception as e:
            error_msg = f"Failed to download: {img_url} | Error: {e}"
            failed_downloads.append(error_msg)
            
    if failed_downloads:
        with open(log_path, "w") as log_file: log_file.write("\n".join(failed_downloads))
    
    return log_path if failed_downloads else None

def scrape_text(url):
    """Scrapes and cleans plain text content from a page using requests/BeautifulSoup."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.6778.265 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove irrelevant elements
        for element in soup(["script", "style", "nav", "header", "footer", "form"]): element.decompose()
        
        raw_text = soup.body.get_text()
        
        # Clean up whitespace and line breaks
        clean_text = re.sub(r'[\r\n]+', '\n', raw_text)
        clean_text = re.sub(r'[ \t]+', ' ', clean_text)
        lines = [line.strip() for line in clean_text.split('\n') if line.strip() and len(line.strip()) > 3]
        final_text = "\n\n".join(lines).strip()
        
        formatted_output = f"""[INFO] Initiating Secure Connection to: {url}\n[STATUS] Analyzing DOM structure for content extraction.\n\nWeb Data Scraper v3.1: Parsing content blocks...\n\n{final_text}\n\n[STATUS] Cleaning and filtering noise from data streams.\n[SUCCESS] Data stream finalized. Content extraction complete.\n[REPORT] {len(lines)} content blocks extracted.\n"""
        return formatted_output
    except requests.exceptions.RequestException as e: 
        return f"[ERROR] Connection Failed or Invalid URL: {e}"
    except Exception as e: 
        return f"[ERROR] An unexpected error occurred: {e}"
# ===========================================================================

import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin

def get_contact_data(url):
    """Extracts phone numbers and emails using Regex from a given URL."""
    result = {'source_url': url, 'contact_page': None, 'emails': [], 'gmail_only': [], 'phones': []}
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                          'AppleWebKit/537.36 (KHTML, like Gecko) '
                          'Chrome/131.0.6778.265 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        page_content = soup.get_text() + str(soup.find_all('a'))

        base_url = f"{urlparse(url).scheme}://{urlparse(url).netloc}"

        # Find potential contact/about/support page
        for a in soup.find_all('a', href=True):
            href = a['href'].lower()
            if any(word in href for word in ['contact', 'about', 'support']):
                result['contact_page'] = urljoin(base_url, a['href'])
                break

        # Regex patterns (non-capturing groups to fix tuple issue)
        EMAIL_REGEX = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        PHONE_REGEX = r'(?:\+\d{1,3}\s*)?(?:\(?\d{2,4}\)?[\s.-]*)?\d{3,4}[\s.-]?\d{3,4}'

        # Extract all matches
        all_emails = set(re.findall(EMAIL_REGEX, page_content, re.IGNORECASE))
        all_phones = set(re.findall(PHONE_REGEX, page_content))

        # Classify emails
        result['emails'] = sorted({e for e in all_emails if 'gmail' not in e.lower() and 'google' not in e.lower()})
        result['gmail_only'] = sorted({e for e in all_emails if 'gmail' in e.lower() or 'google' in e.lower()})

        # Clean phone numbers
        cleaned_phones = []
        for p in all_phones:
            p_clean = re.sub(r'[^\d+]', ' ', p).strip()
            if len(p_clean.replace(' ', '')) >= 7:
                cleaned_phones.append(p_clean)
        result['phones'] = sorted(set(cleaned_phones))

        return result

    except requests.exceptions.RequestException as e:
        return {"error": f"CONNECTION ERROR: Could not reach the server or invalid URL. Details: {e}"}
    except Exception as e:
        return {"error": f"EXTRACTION ERROR: Unexpected issue occurred. Details: {e}"}

# ==============================================================================
# 2. FLASK ROUTES (NEW AND EXISTING)
# ==============================================================================

# --- Existing Template Routes ---
@app.route('/')
def index(): return render_template('landing.html')
@app.route('/nextpage')
def nextpage(): return render_template('nextpage.html')
@app.route('/image')
def Get_image(): return render_template('image.html')
@app.route('/text_scraper') 
def text_scraper_page(): return render_template('text_scraper.html')
@app.route('/contact_extractor')
def contact_extractor_page(): return render_template('contact_extractor.html')
@app.route('/all_scarper')
def all_scarper(): return render_template('all_scraper.html')
@app.route('/cd')
def codroid(): return render_template('codroidhub.html')
@app.route('/credit')
def credit(): return render_template('credits.html')
@app.route('/info')
def info(): return render_template('info.html')

# --- NEW COMPREHENSIVE SCAN ROUTE (/scrape_all) ---
@app.route('/scrape_all', methods=['POST'])
def scrape_all():
    url = request.json.get('url')
    if not url:
        return jsonify({"error": "URL not provided."}), 400

    # 1. Contact Extraction (call helper, not route)
    contact_data = get_contact_data(url)
    if isinstance(contact_data, dict) and contact_data.get('error'):
        return jsonify({"error": contact_data['error']}), 500

    # 2. Text Extraction
    text_content = scrape_text(url)
    if text_content.startswith("[ERROR]"):
        return jsonify({"error": text_content}), 500

    # 3. Image Scraping (only get the list of URLs)
    try:
        image_df = scrape_images(url)
        image_list = image_df['src'].tolist() if not image_df.empty else []
    except Exception as e:
        print(f"Image scraping partial failure: {e}")
        image_list = []

    # Combine results
    response_data = {
        'contact_page': contact_data.get('contact_page'),
        'emails': contact_data.get('emails', []),
        'gmail_only': contact_data.get('gmail_only', []),
        'phones': contact_data.get('phones', []),
        'text': text_content,
        'images': image_list
    }

    return jsonify(response_data)


# --- Existing Download and Scrape Routes (Kept for compatibility) ---

@app.route('/get_images', methods=['POST'])
def get_images():
    url = request.json.get('url')
    try:
        df = scrape_images(url)
        if df.empty: return jsonify({"images": [], "error": "No images found after scraping and preliminary filtering."})
        csv_path = os.path.join(DOWNLOAD_FOLDER, "image_data.csv")
        df.to_csv(csv_path, index=False)
        images_folder = os.path.join(DOWNLOAD_FOLDER, "images")
        log_file_path = download_images(df, images_folder)
        image_urls = df['src'].tolist()
        response = {"images": image_urls, "csv": "/download/csv", "images_zip": "/download/images"}
        if log_file_path: response["log"] = "/download/log"
        return jsonify(response)
    except Exception as e: print(f"Global scraping error: {e}"); return jsonify({"images": [], "error": str(e)})

@app.route('/download/csv')
def download_csv():
    csv_path = os.path.join(DOWNLOAD_FOLDER, "image_data.csv"); return send_file(csv_path, as_attachment=True, download_name="image_data.csv")
@app.route('/download/log')
def download_log():
    log_path = os.path.join(DOWNLOAD_FOLDER, "images", "failed_downloads.log")
    if not os.path.exists(log_path): return jsonify({"error": "Log file not found."}), 404
    return send_file(log_path, as_attachment=True, download_name="failed_downloads.log")
@app.route('/download/images')
def download_images_zip():
    memory_file = BytesIO(); zipf = zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED); images_folder = os.path.join(DOWNLOAD_FOLDER, "images")
    for root, _, files in os.walk(images_folder):
        for file in files:
            if file == "failed_downloads.log": continue
            zipf.write(os.path.join(root, file), arcname=file)
    zipf.close(); memory_file.seek(0); return send_file(memory_file, download_name='images.zip', as_attachment=True)

@app.route('/get_text', methods=['POST'])
def get_text():
    url = request.json.get('url')
    if not url: return jsonify({"error": "URL not provided."}), 400
    scraped_data = scrape_text(url)
    if scraped_data.startswith("[ERROR]"): return jsonify({"text": scraped_data, "error": True})
    file_path = os.path.join(DOWNLOAD_FOLDER, "scraped_data.txt")
    with open(file_path, "w", encoding="utf-8") as f: f.write(scraped_data.strip())
    return jsonify({"text": scraped_data, "download_url": "/download/text"})
@app.route('/download/text')
def download_text():
    file_path = os.path.join(DOWNLOAD_FOLDER, "scraped_data.txt")
    if not os.path.exists(file_path): return jsonify({"error": "File not found. Please scrape data first."}), 404
    return send_file(file_path, as_attachment=True, download_name="scraped_data.txt", mimetype='text/plain')

@app.route('/get_contact_info', methods=['POST'])
def get_contact_info():
    url = request.json.get('url')
    if not url: return jsonify({"error": "URL not provided."}), 400
    data = get_contact_data(url)
    if data.get('error'): return jsonify(data), 500
    contacts_list = []
    for email in data['emails']: contacts_list.append({'Type': 'Email', 'Value': email, 'Category': 'General'})
    for g_email in data['gmail_only']: contacts_list.append({'Type': 'Email', 'Value': g_email, 'Category': 'Gmail/Personal'})
    for phone in data['phones']: contacts_list.append({'Type': 'Phone', 'Value': phone, 'Category': 'Direct Line'})
    df = pd.DataFrame(contacts_list); csv_path = os.path.join(DOWNLOAD_FOLDER, "extracted_contacts.csv"); df.to_csv(csv_path, index=False)
    return jsonify(data)
@app.route('/download/contacts_csv')
def download_contacts_csv():
    csv_path = os.path.join(DOWNLOAD_FOLDER, "extracted_contacts.csv")
    if not os.path.exists(csv_path): return jsonify({"error": "Contact data not found. Please run the extractor first."}), 404
    return send_file(csv_path, as_attachment=True, download_name="extracted_contacts.csv")

# Create upload folder if doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)



if __name__ == '__main__':
    app.run(debug=True)