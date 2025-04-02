import streamlit as st
import torch
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from ultralytics import YOLO
import io
import base64

# Background Image Path
background_image_path = "C:\\Users\\User\\Music\\streamapp\\backgrounds\\background1.png"

def get_base64_of_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

# Custom CSS for UI
base64_image = get_base64_of_image(background_image_path)
st.markdown(f"""
    <style>
        .stApp {{
            background-image: url("data:image/png;base64,{base64_image}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            text-align: center;
            display: flex;
            flex-direction: column;
            justify-content: center;
            height: 100vh;
        }}
        .main-title {{
            padding-top:100px;
            font-size: 50px;
            font-weight: bold;
            color: white;
            text-transform: uppercase;
            text-shadow: 2px 2px 5px black;
            margin-bottom: 20px;
        }}
        .subtext {{
            font-size: 20px;
            color: white;
            text-shadow: 1px 1px 3px black;
            margin-bottom: 40px;
        }}
        .button-container {{
            display: flex;
            justify-content: center;
        }}
        .stButton>button {{
            background-color: #1e1e1e;
            color: #ffffff;
            font-size: 18px;
            padding: 12px 32px;
            border-radius: 6px;
            font-weight: 600;
            border: 2px solid #ffffff;
            transition: all 0.3s ease-in-out;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
        }}
        .stButton>button:hover {{
            background-color: #ffffff;
            color: #000000;
            border: 2px solid #000000;
            transform: scale(1.05);
        }}
    </style>
""", unsafe_allow_html=True)

# Class Names
CLASS_NAMES = {
    0: "Healthy Chilli, ఆరోగ్యంగా ఉన్న మిరప",
    1: "Potato Common Scab (Fruit), బంగాళాదుంప సాధారణ దద్దుర్లు (పండు)",
    2: "Eggplant Healthy (Fruit), ఆరోగ్యంగా ఉన్న వంకాయ (పండు)",
    3: "Eggplant Healthy (Leaf), ఆరోగ్యంగా ఉన్న వంకాయ (ఆకు)",
    4: "Chilli Bacterial Leaf Spot, మిరప బాక్టీరియల్ ఆకు మచ్చలు",
    5: "Eggplant Fruit Rot, వంకాయ పండు కుళ్ళిపోవడం",
    6: "Potato Alternaria Solani (Leaf), బంగాళాదుంప ఆల్టర్నేరియా సోలాని (ఆకు)",
    7: "Chilli Mosaic Leaf Virus, మిరప మోజాయిక్ ఆకు వైరస్",
    8: "Potato Phytophthora Infestans (Leaf), బంగాళాదుంప ఫైటోఫ్తోరా ఇన్‌ఫెస్టాన్స్ (ఆకు)",
    9: "Potato Healthy (Fruit), ఆరోగ్యంగా ఉన్న బంగాళాదుంప (పండు)",
    10: "Potato Healthy (Leaf), ఆరోగ్యంగా ఉన్న బంగాళాదుంప (ఆకు)",
    11: "Tomato Late Blight (Leaf), టొమాటో లేట్ బ్లైట్ (ఆకు)",
    12: "Tomato Anthracnose, టొమాటో యాంథ్రాక్నోజ్",
    13: "Eggplant Colorado Potato Beetle, వంకాయ కలరాడో బంగాళాదుంప పురుగు",
    14: "Chilli Anthracnose, మిరప యాంథ్రాక్నోజ్",
    15: "Eggplant Cercospora Leaf Spot, వంకాయ సర్కోస్పోరా ఆకు మచ్చ",
    16: "Healthy Chilli (Leaf), ఆరోగ్యంగా ఉన్న మిరప (ఆకు)",
    17: "Tomato Healthy, ఆరోగ్యంగా ఉన్న టొమాటో",
    18: "Tomato Bacterial Spot, టొమాటో బాక్టీరియల్ మచ్చలు",
    19: "Eggplant Fruit Rot, వంకాయ పండు కుళ్ళిపోవడం",
}

# YOLO Model Paths
MODEL_PATHS = {
    "YOLOv8": r"C:\\Users\\User\\Music\\streamapp\\models\\v8.pt",
    "YOLOv11": r"C:\\Users\\User\\Music\\streamapp\\models\\v11.pt",
    "YOLOv12": r"C:\\Users\\User\\Music\\streamapp\\models\\v12_ft.pt"
}

@st.cache_resource
def load_model(weights_path):
    model = YOLO(weights_path)
    return model

def predict(model, image):
    results = model(image)
    return results

def draw_boxes(image, results):
    image_np = np.array(image)
    detected_data = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            class_name_full = CLASS_NAMES.get(cls, f"Class {cls}")
            eng_name, tel_name = class_name_full.split(",")
            detected_data.append((eng_name.strip(), tel_name.strip(), conf))

            label = f"{eng_name.strip()}: {conf:.2f}"
            cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
            font_scale = 0.4
            thickness = 1
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            cv2.rectangle(image_np, (x1, y1 - h - 5), (x1 + w, y1), (0, 255, 0), -1)
            cv2.putText(image_np, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)

    return Image.fromarray(image_np), detected_data

def generate_report(detection_results):
    report = io.StringIO()
    report.write("YOLO Object Detection Report\n")
    report.write("=" * 30 + "\n\n")
    for img_name, diseases in detection_results.items():
        report.write(f"Image: {img_name}\n")
        if diseases:
            report.write(f"Detected Diseases: {', '.join([f'{d[0]} / {d[1]} (Conf: {d[2]:.2f})' for d in diseases])}\n")
        else:
            report.write("No diseases detected.\n")
        report.write("-" * 30 + "\n")
    return report.getvalue()

def generate_csv(detection_results):
    csv_data = io.StringIO()
    data = []
    for img_name, diseases in detection_results.items():
        for eng, tel, confidence in diseases:
            data.append({"Image Name": img_name, "Detected Disease (EN)": eng, "Detected Disease (TE)": tel, "Confidence Score": confidence})
    df = pd.DataFrame(data)
    df.to_csv(csv_data, index=False)
    return csv_data.getvalue()

def set_page(page_name):
    st.session_state.page = page_name

# --- App Logic ---
if "page" not in st.session_state:
    st.session_state.page = "home"

if st.session_state.page == "home":
    st.markdown("<div class='main-title'>CROP DISEASE DETECTION</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtext'>Detect crop diseases with AI-powered YOLO models</div>", unsafe_allow_html=True)
    st.markdown("<div class='button-container'>", unsafe_allow_html=True)
    if st.button("GET STARTED"):
        set_page("detection")
    st.markdown("</div>", unsafe_allow_html=True)

elif st.session_state.page == "detection":
    st.markdown("<div class='main-title'>CROP DISEASE DETECTION</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtext'>Upload an image to detect diseases in crops</div>", unsafe_allow_html=True)

    selected_model = st.radio("Select YOLO Version:", ["YOLOv8", "YOLOv11", "YOLOv12"], index=1)
    model = load_model(MODEL_PATHS[selected_model])

    uploaded_files = st.file_uploader("\U0001F4E4 Upload Images", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

    detection_results = {}
    st.markdown("<div class='button-container'>", unsafe_allow_html=True)
    detect_clicked = st.button("\U0001F50D Detect Disease")
    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded_files and detect_clicked:
        cols = st.columns(4)
        for i, uploaded_file in enumerate(uploaded_files):
            image = Image.open(uploaded_file).convert("RGB")
            results = predict(model, image)
            image_with_boxes, detected_data = draw_boxes(image, results)

            with cols[i % 4]:
                st.image(image_with_boxes, use_container_width=True)
                if detected_data:
                    st.markdown("<div style='font-weight:bold;'>Detected:</div>", unsafe_allow_html=True)
                    for eng, tel, conf in detected_data:
                        st.write(f"- {eng} / {tel} ({conf:.2f})")
                else:
                    st.write("No diseases detected.")

            detection_results[uploaded_file.name] = detected_data

        report_text = generate_report(detection_results)
        st.download_button("Download Report (TXT)", report_text, file_name="detection_report.txt", mime="text/plain")

        csv_data = generate_csv(detection_results)
        st.download_button("Download Report (CSV)", csv_data, file_name="detection_report.csv", mime="text/csv")
