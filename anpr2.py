import io
import json
import torch
import cv2
import matplotlib.pyplot as plt
import pytesseract        # added code
from pytesseract import image_to_string # added code 
import glob
import pandas as pd
import re
import streamlit as st
import os



pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
model = torch.hub.load("D:/pricex_model_api/deep_learning/yolov5-master", 'custom', path = "D:/pricex_model_api/deep_learning/best.pt", source='local', force_reload=True)
upload_path = "uploads/"


@st.cache(persist=True,allow_output_mutation=True,show_spinner=False,suppress_st_warning=True)
def yolomodel(img):
	
	frame = cv2.imread(img)
	detections = model(frame)
	results = detections.pandas().xyxy[0].to_dict(orient="records")        
	for result in results:
		con = result['confidence']
		cs = result['class']
		x1 = int(result['xmin'])
		y1 = int(result['ymin'])
		x2 = int(result['xmax'])
		y2 = int(result['ymax'])
		cropped_img = frame[y1:y2,x1:x2]

		try:
			text = pytesseract.image_to_string(cropped_img, config='-l eng --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890') # added code
			raw = re.sub("[^A-Z0-9 -]", "", text)
			final_result = re.sub(r'[\W_]+', '', raw)
			return final_result

		except:
			return f"Not Readable"




st.title("Automatic Number Plate Detection")
st.info('Supports all popular image formats 📷 - PNG, JPG, JPEG')
uploaded_file = st.file_uploader("Upload Image of car's number plate 🚓", type=["png","jpg","bmp","jpeg","jfif"])


if uploaded_file is not None:
    with open(os.path.join(upload_path,uploaded_file.name),"wb") as f:
        f.write((uploaded_file).getbuffer())
    
    with st.spinner(f"Working... 💫"):
        uploaded_image = os.path.abspath(os.path.join(upload_path,uploaded_file.name))


if st.button('Get the Number Plate'):
	results = yolomodel(uploaded_image)
	st.subheader(f"Number Plate: {results}")