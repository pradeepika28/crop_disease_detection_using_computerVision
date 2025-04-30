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
background_image_path = "images/background1.png"

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
    0: "Healthy Chilli",
    1: "Potato Common Scab (Fruit)",
    2: "Eggplant Healthy (Fruit)",
    3: "Eggplant Healthy (Leaf)",
    4: "Chilli Bacterial Leaf Spot",
    5: "Eggplant Fruit Rot",
    6: "Potato Alternaria Solani (Leaf)",
    7: "Chilli Mosaic Leaf Virus",
    8: "Potato Phytophthora Infestans (Leaf)",
    9: "Potato Healthy (Fruit)",
    10: "Potato Healthy (Leaf)",
    11: "Tomato Late Blight (Leaf)",
    12: "Tomato Anthracnose",
    13: "Eggplant Colorado Potato Beetle",
    14: "Chilli Anthracnose",
    15: "Eggplant Cercospora Leaf Spot",
    16: "Healthy Chilli (Leaf)",
    17: "Tomato Healthy",
    18: "Tomato Bacterial Spot",
    19: "Eggplant Fruit Rot",
}

# Prescriptions
PRESCRIPTIONS = {
    "Healthy Chilli": "No treatment needed. Keep monitoring and  Maintain optimal conditions.",
    "Potato Common Scab (Fruit)": "Grow scab-resistant varieties like ‘Russet Burbank’ or ‘Superior’ and use certified disease-free seed tubers. Maintain a slightly acidic soil pH (5.0–5.2) and consistent moisture, especially during early tuber formation. Rotate crops with non-host plants, incorporate organic matter, and apply sulfur or specific bactericides in severe cases.",
    "Eggplant Healthy (Fruit)": "No treatment needed. Keep monitoring and  Maintain optimal conditions.",
    "Eggplant Healthy (Leaf)": "No treatment needed. Keep monitoring and  Maintain optimal conditions.",
    "Chilli Bacterial Leaf Spot": "Use resistant or tolerant chili varieties and treat seeds with hot water or streptocycline before sowing. Practice crop sanitation by removing infected debris and apply copper-based bactericides with streptocycline at 10–12 day intervals. Use drip irrigation to minimize leaf wetness, maintain wide plant spacing for better airflow, and rotate with non-solanaceous crops to reduce bacterial load in the soil.",
    "Eggplant Fruit Rot": "Grow resistant or tolerant eggplant varieties to reduce fruit rot risk. Apply fungicides like carbendazim or mancozeb at 10–15 day intervals during wet conditions, and rotate fungicides to prevent resistance. Rotate crops with non-host plants, use drip irrigation, and ensure good drainage. Practice sanitation by removing infected fruits and debris, prune lower branches for better airflow, and apply mulch to prevent soil splashing. Maintain weed-free fields to reduce disease pressure.",
    "Potato Alternaria Solani (Leaf)": "Use resistant or tolerant potato varieties to reduce disease risk. Apply fungicides like chlorothalonil or mancozeb, and systemic fungicides like azoxystrobin, especially during humid conditions. Maintain proper plant spacing for airflow, avoid overhead irrigation, and rotate crops with non-host plants. Practice field sanitation by removing infected plants and debris, and prune heavily infected leaves to improve air circulation.",
    "Chilli Mosaic Leaf Virus": "Use mosaic virus-resistant or tolerant chili varieties and treat seeds with trisodium phosphate to remove surface viruses. Control aphids with neem oil or systemic insecticides, and regularly remove infected plants (rogueing) to prevent spread. Practice crop rotation with non-host crops, maintain field hygiene, and plant maize or sorghum as barrier crops to reduce aphid movement. Avoid mechanical transmission by minimizing plant handling when wet and disinfecting tools regularly.",
    "Potato Phytophthora Infestans (Leaf)": "Plant late blight-resistant potato varieties and apply fungicides like mancozeb, chlorothalonil, or metalaxyl regularly, rotating them to prevent resistance. Monitor crops closely after rain, use drip irrigation, maintain good plant spacing, and practice field sanitation by removing infected plants and debris. Rotate with non-host crops like cereals for 2–3 years to lower pathogen levels.",
    "Potato Healthy (Fruit)": "No treatment needed. Keep monitoring and  Maintain optimal conditions.",
    "Potato Healthy (Leaf)": "No treatment needed. Keep monitoring and  Maintain optimal conditions.",
    "Tomato Late Blight (Leaf)": "Plant resistant tomato varieties and apply systemic fungicides like chlorothalonil, metalaxyl, or mancozeb regularly, rotating them to prevent resistance. Rotate with non-host crops, use drip irrigation, remove infected debris, prune affected parts for better airflow, and maintain proper plant spacing to reduce humidity and disease spread.",
    "Tomato Anthracnose": "Plant anthracnose-resistant tomato varieties and rotate with non-host crops like cereals or legumes to lower fungal load. Use drip irrigation to keep foliage dry, apply fungicides like copper-based products or azoxystrobin during humid periods, and maintain field sanitation by removing infected debris. Prune infected parts and ensure proper plant spacing to improve airflow and reduce humidity.",
    "Eggplant Colorado Potato Beetle": "Plant pest-tolerant potato and eggplant varieties, handpick beetles and larvae early, and apply insecticides like spinosad or neem-based products while rotating chemicals to prevent resistance. Rotate crops with non-hosts, use trap crops, encourage natural predators, and maintain field sanitation by removing residues and volunteer plants.",
    "Chilli Anthracnose": "Grow anthracnose-resistant chili varieties, treat seeds with carbendazim, and apply fungicides like carbendazim or mancozeb at regular intervals, rotating them to prevent resistance. Remove infected fruits and residues, rotate with non-host crops, ensure good drainage, and maintain proper plant spacing for better airflow.",
    "Eggplant Cercospora Leaf Spot": "Use Cercospora-resistant eggplant varieties, apply fungicides like mancozeb or copper oxychloride at regular intervals, and rotate crops with non-hosts like cereals. Prefer drip irrigation, remove infected debris, prune dense canopies for better airflow, and maintain proper spacing and drainage to reduce humidity and disease spread.",
    "Healthy Chilli (Leaf)": "No treatment needed. Keep monitoring and  Maintain optimal conditions.",
    "Tomato Healthy": "No treatment needed. Keep monitoring and  Maintain optimal conditions.",
    "Tomato Bacterial Spot": "Rotate tomatoes with non-host crops, use resistant varieties, and prefer drip irrigation to reduce bacterial spread. Remove infected debris, apply copper-based bactericides preventively, prune infected leaves, and maintain proper spacing to improve airflow and lower humidity.",
    "Eggplant Fruit Rot": "Grow resistant eggplant varieties, rotate with non-host crops, and use drip irrigation to prevent fruit rot. Apply fungicides like carbendazim or mancozeb at intervals, prune lower branches, maintain field sanitation, and mulch to control moisture and soil splashing.",
}

# Model Path
YOLOV12_MODEL_PATH = "models/v12_ft.pt"

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
            class_name = CLASS_NAMES.get(cls, f"Class {cls}")
            detected_data.append((class_name, conf))

            label = f"{class_name}: {conf:.2f}"
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
            for d in diseases:
                report.write(f"Detected: {d[0]} (Confidence: {d[1]:.2f})\n")
                prescription = PRESCRIPTIONS.get(d[0], "No prescription available.")
                report.write(f"Prescription: {prescription}\n")
        else:
            report.write("No diseases detected.\n")
        report.write("-" * 30 + "\n")
    return report.getvalue()

def generate_csv(detection_results):
    csv_data = io.StringIO()
    data = []
    for img_name, diseases in detection_results.items():
        for name, confidence in diseases:
            prescription = PRESCRIPTIONS.get(name, "No prescription available.")
            data.append({
                "Image Name": img_name,
                "Detected Disease": name,
                "Confidence Score": confidence,
                "Prescription": prescription
            })
    df = pd.DataFrame(data)
    df.to_csv(csv_data, index=False)
    return csv_data.getvalue()

def set_page(page_name):
    st.session_state.page = page_name


if "page" not in st.session_state:
    st.session_state.page = "home"

model = load_model(YOLOV12_MODEL_PATH)

if st.session_state.page == "home":
    st.markdown("<div class='main-title'>CROP DISEASE DETECTION</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtext'>Detect crop diseases with AI-powered YOLOv12 model</div>", unsafe_allow_html=True)
    st.markdown("<div class='button-container'>", unsafe_allow_html=True)
    if st.button("GET STARTED"):
        set_page("detection")
    st.markdown("</div>", unsafe_allow_html=True)

elif st.session_state.page == "detection":
    st.markdown("<div class='main-title'>CROP DISEASE DETECTION</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtext'>Upload an image to detect diseases in crops</div>", unsafe_allow_html=True)

    uploaded_files = st.file_uploader("\U0001F4E4 Upload Images", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

    detection_results = {}
    st.markdown("<div class='button-container'>", unsafe_allow_html=True)
    detect_clicked = st.button("\U0001F50D Detect Disease")
    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded_files and detect_clicked:
        cols = st.columns(2)  
        for i, uploaded_file in enumerate(uploaded_files):
            image = Image.open(uploaded_file).convert("RGB")
            results = predict(model, image)
            image_with_boxes, detected_data = draw_boxes(image, results)

            with cols[i % 2]:
                st.image(image_with_boxes, use_container_width=True)
                if detected_data:
                    st.markdown("<div style='font-weight:bold; color:white;'>Detected:</div>", unsafe_allow_html=True)
                    for name, conf in detected_data:
                        st.markdown(
                            f"<div style='color: white; font-size: 16px;'>{name} ({conf:.2f})</div>",
                            unsafe_allow_html=True
                        )
                        prescription = PRESCRIPTIONS.get(name, "No prescription available.")
                        st.markdown(
                            f"<div style='color: white; font-size: 14px; margin-bottom:10px;'>💊 Prescription: {prescription}</div>",
                            unsafe_allow_html=True
                        )
                else:
                    st.markdown("<div style='color:white;'>No diseases detected.</div>", unsafe_allow_html=True)

            detection_results[uploaded_file.name] = detected_data

        report_text = generate_report(detection_results)
        st.download_button("Download Report (TXT)", report_text, file_name="detection_report.txt", mime="text/plain")

        csv_data = generate_csv(detection_results)
        st.download_button("Download Report (CSV)", csv_data, file_name="detection_report.csv", mime="text/csv")
