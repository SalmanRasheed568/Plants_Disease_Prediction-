Plant Disease Classification Project
=====================================

Project Description:
---------------------
This project is designed to detect and classify plant diseases using deep learning technique (CNN).
It includes a Streamlit application with the following features:
- AI Engine: Predict diseases based on plant images.
- Supplements: Information and remedies for plant diseases.
- Description about the diseases
- Prevention tips 
- confidences score
- Alternative Prediction levels.


Directory Structure:
---------------------
BCS-OPEN-12/
│
├── src/
│   ├── main.py,CNN,Disease_info.csv,supplement_info.csv 
│   ├── models/
│   ├── assets/Test_Images,temp 
│   └── (other supporting Python files)
│
├──requirements.txt 
|--README.txt

Instructions to Run:
---------------------
1. Install Python 3.8 or above.
2. install pychram
3. Install required packages by running:
   pip install -r requirements.txt

   (If 'requirements.txt' is not available, manually install main libraries like Streamlit, Tensorflow/Keras,pytorch, OpenCV, etc.)

3. To run the Streamlit application, open your terminal, navigate to the 'src' folder, and type:
   streamlit run main.py

4. The app will open automatically in your default web browser.

5. If there is any issue with the main.py or any other file you can download it from the Github link or run it directly from the Github repository. 
6. Github link:

Requirements:
--------------
aiohappyeyeballs==2.4.6
aiohttp==3.11.12
aiosignal==1.3.2
altair==5.5.0
attrs==25.1.0
beautifulsoup4==4.13.3
blinker==1.9.0
cachetools==5.5.1
certifi==2025.1.31
chardet==3.0.4
charset-normalizer==3.4.1
click==8.1.8
colorama==0.4.6
filelock==3.17.0
frozenlist==1.5.0
fsspec==2025.2.0
gdown==5.2.0
gitdb==4.0.12
GitPython==3.1.44
googletrans==4.0.0rc1
gunicorn==23.0.0
h11==0.9.0
h2==3.2.0
hpack==3.0.0
hstspreload==2025.1.1
httpcore==0.9.1
httpx==0.13.3
hyperframe==5.2.0
idna==2.10
itsdangerous==2.2.0
Jinja2==3.1.5
jsonschema==4.23.0
jsonschema-specifications==2024.10.1
markdown-it-py==3.0.0
MarkupSafe==3.0.2
mdurl==0.1.2
mpmath==1.3.0
multidict==6.1.0
narwhals==1.25.1
networkx==3.4.2
numpy==2.2.2
openai==0.28.1
packaging==24.2
pandas==2.2.3
pillow==11.1.0
plotly==6.0.0
propcache==0.2.1
protobuf==5.29.3
pyarrow==19.0.0
pydeck==0.9.1
Pygments==2.19.1
PySocks==1.7.1
python-dateutil==2.9.0.post0
pytz==2025.1
referencing==0.36.2
requests==2.32.3
rfc3986==1.5.0
rich==13.9.4
rpds-py==0.22.3
setuptools==75.8.0
six==1.17.0
smmap==5.0.2
sniffio==1.3.1
soupsieve==2.6
streamlit==1.42.0
sympy==1.13.1
tenacity==9.0.0
toml==0.10.2
torch==2.6.0
torchaudio==2.6.0
torchvision==0.21.0
tornado==6.4.2
tqdm==4.67.1
typing_extensions==4.12.2
tzdata==2025.1
urllib3==2.3.0
watchdog==6.0.0
Werkzeug==3.1.3
yarl==1.18.3


Important Notes:
----------------
- Make sure you have a stable internet connection bcz you have to download the model from the drive at runtime.
- Because as per instructions the train model size is large so we didn't add it here.
- If you face issues related to ports (e.g., port 8501 busy), you can specify another port:
  streamlit run app.py --server.port 8502

Contact:
---------
For queries or issues, please contact the project team.


