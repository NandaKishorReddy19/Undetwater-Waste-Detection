import streamlit as st
import os
import cv2
from ultralytics import YOLO
from PIL import Image
import tempfile


model = YOLO("C:/Users/nanda/OneDrive/Desktop/MYproject/runs/detect/train2/weights/best.pt") 

# Function to process image
def process_image(image_path):
    results = model.predict(source=image_path, save=False)
    annotated_image = results[0].plot()
    output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.png').name
    Image.fromarray(annotated_image).save(output_path)
    return output_path

# Function to process video
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    out = cv2.VideoWriter(temp_output.name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(source=frame, save=False)
        annotated_frame = results[0].plot()
        out.write(annotated_frame)

    cap.release()
    out.release()

    return temp_output.name


st.title("Underwater Waste Detection Web Service")
st.write("Upload an image or video file (JPEG, JPG, PNG, MP4) to perform object detection.")

uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png", "mp4"])

if uploaded_file:
    file_extension = uploaded_file.name.split('.')[-1].lower()
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}')
    temp_file.write(uploaded_file.read())
    temp_file.close()

    if file_extension in ["jpg", "jpeg", "png"]:
        st.write("Processing image...")
        st.write("Running the model...")
        output_image_path = process_image(temp_file.name)
        st.image(output_image_path, caption="Processed Image", use_column_width=True)
        with open(output_image_path, "rb") as file:
            st.download_button(label="Download Processed Image", data=file, file_name="processed_image.png", mime="image/png")

    elif file_extension == "mp4":
        st.write("Processing video...")
        st.write("Running the model...")
        output_video_path = process_video(temp_file.name)
        st.video(output_video_path)
        with open(output_video_path, "rb") as file:
            st.download_button(label="Download Processed Video", data=file, file_name="processed_video.mp4", mime="video/mp4")

    else:
        st.error("Unsupported file type! Please upload a JPEG, JPG, PNG, or MP4 file.")

    os.remove(temp_file.name)
