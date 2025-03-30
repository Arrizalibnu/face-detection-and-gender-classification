import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

model = YOLO("face_detection.pt")

def detect_faces(image):
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = model(img)[0]

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        label = "Female" if cls == 0 else "Male"
        color = (245, 0, 255) if cls == 0 else (255, 5, 0)
        text_color = (0, 0, 0)
        box_height = y2 - y1
        font_size = max(0.5, box_height * 0.002)
        thickness = max(1, int(box_height * 0.002))
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_size, thickness)[0]
        text_x, text_y = x1 + 5, y1 + 15  
        bg_x1, bg_y1 = text_x - 5, text_y - text_size[1] - 5
        bg_x2, bg_y2 = text_x + text_size[0] + 5, text_y + 5

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(img, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)
        cv2.putText(img, label, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_size, text_color, thickness)

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

st.title("Face & Gender Detection")
st.write("Upload your photo to detect face and gender.")

uploaded_file = st.file_uploader("Choose your image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Image uploaded", use_column_width=True)

    with st.spinner("Loading..."):
        result_img = detect_faces(image)

    st.image(result_img, caption="Result detection", use_column_width=True)

st.write("Made by Streamlit & YOLO")