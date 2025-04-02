import streamlit as st
import torch
import cv2
import numpy as np
import pandas as pd  # For CSV export
from PIL import Image
from ultralytics import YOLO  # Make sure ultralytics is installed: pip install ultralytics
import io  # For file handling

# Define class names based on your dataset
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
    19: "Eggplant Fruit Rot"
}

# Define model paths for different YOLO versions
MODEL_PATHS = {
    "YOLOv8": r"C:\Users\User\Music\streamapp\models\v8.pt",
    "YOLOv11": r"C:\Users\User\Music\streamapp\models\v11.pt",
    "YOLOv12": r"C:\Users\User\Music\streamapp\models\v12.pt"
}

# Load YOLO model based on selection
@st.cache_resource
def load_model(weights_path):
    model = YOLO(weights_path)  # Load YOLO model
    return model

# Perform inference
def predict(model, image):
    results = model(image)  # Run YOLO model on image
    return results

# Draw bounding boxes and extract detected diseases
def draw_boxes(image, results):
    image_np = np.array(image)  # Convert PIL image to NumPy array (OpenCV format)
    detected_data = []  # Store detected disease details (name, confidence score)
    
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # Convert tensor to list
            conf = float(box.conf[0])  # Confidence score
            cls = int(box.cls[0])  # Class ID
            
            # Get the class name from the dictionary
            class_name = CLASS_NAMES.get(cls, f"Class {cls}")  # Default to "Class {id}" if not found
            detected_data.append((class_name, conf))  # Store detected disease with confidence
            
            label = f"{class_name}: {conf:.2f}"

            # Draw bounding box
            cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Label background
            font_scale = 0.4  # Reduced font size
            thickness = 1  # Reduced thickness
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            cv2.rectangle(image_np, (x1, y1 - h - 5), (x1 + w, y1), (0, 255, 0), -1)  # Adjusted label background
            
            # Put label text
            cv2.putText(image_np, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
    
    return Image.fromarray(image_np), detected_data  # Convert back to PIL Image

# Generate a report as a text file
def generate_report(detection_results):
    report = io.StringIO()
    report.write("YOLO Object Detection Report\n")
    report.write("=" * 30 + "\n\n")

    for img_name, diseases in detection_results.items():
        report.write(f"Image: {img_name}\n")
        if diseases:
            report.write(f"Detected Diseases: {', '.join([f'{d[0]} (Conf: {d[1]:.2f})' for d in diseases])}\n")
        else:
            report.write("No diseases detected.\n")
        report.write("-" * 30 + "\n")

    return report.getvalue()

# Generate CSV data for export
def generate_csv(detection_results):
    csv_data = io.StringIO()
    data = []  # List to store dictionary rows

    for img_name, diseases in detection_results.items():
        for disease, confidence in diseases:
            data.append({"Image Name": img_name, "Detected Disease": disease, "Confidence Score": confidence})

    # Create DataFrame and write to CSV
    df = pd.DataFrame(data)
    df.to_csv(csv_data, index=False)
    
    return csv_data.getvalue()


# Streamlit UI
def main():
    st.title("YOLO Object Detection for Crop Diseases")

    # Model selection using radio button
    selected_model = st.radio("Select YOLO Version:", ["YOLOv8", "YOLOv11", "YOLOv12"], index=1)

    # Load the selected model
    model = load_model(MODEL_PATHS[selected_model])

    # File uploader (multiple files allowed)
    uploaded_files = st.file_uploader("Upload Images", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

    detection_results = {}  # Store results for report generation

    # **Submit button to trigger detection**
    if uploaded_files and st.button("Submit"):
        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file).convert("RGB")
            results = predict(model, image)
            image_with_boxes, detected_data = draw_boxes(image, results)
            
            # Display image with detections
            st.image(image_with_boxes, caption=f"Detected Diseases ({selected_model})", use_container_width=True)

            # Store results for report
            detection_results[uploaded_file.name] = detected_data

            # Display detected diseases below the image
            if detected_data:
                st.subheader("Detected Diseases:")
                for disease, conf in detected_data:
                    st.write(f"- **{disease}** (Confidence: {conf:.2f})")
            else:
                st.subheader("No diseases detected.")

        # Generate and provide download button for **text report**
        report_text = generate_report(detection_results)
        st.download_button("Download Report (TXT)", report_text, file_name="detection_report.txt", mime="text/plain")

        # Generate and provide download button for **CSV file**
        csv_data = generate_csv(detection_results)
        st.download_button("Download Report (CSV)", csv_data, file_name="detection_report.csv", mime="text/csv")

if __name__ == "__main__":
    main()