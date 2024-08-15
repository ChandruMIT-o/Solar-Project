import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import os

# Load the models once
model_cls = YOLO("custom_models/best_30e_cls.pt")
model_seg = YOLO("custom_models/best_30e_seg.pt")
model_det = YOLO("custom_models/last_30e_det.pt")

def process_image(image):
    # Perform detection
    results_det = model_det(image)
    det_image = None
    seg_image = None
    cls_image = None

    for result in results_det:
        original_image = result.orig_img
        boxes = result.boxes.xyxy

        for bb in boxes:
            x, y, w, h = bb
            x, y, w, h = int(x), int(y), int(w), int(h)
            det_image = original_image[y:h, x:w]

            # Convert to PIL image
            det_pil_image = Image.fromarray(det_image)

            # Perform segmentation
            seg_results = model_seg(det_pil_image)
            for seg_result in seg_results:
                seg_image = seg_result.plot()  # Getting the segmented image

            # Perform classification
            cls_results = model_cls(det_pil_image)
            for cls_result in cls_results:
                cls_image = cls_result.plot()  # Getting the classified image

    return original_image, det_image, seg_image, cls_image

# Streamlit UI
st.title("Image Detection, Segmentation, and Classification")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the file to an open CV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

    # Process the image
    original_image, det_image, seg_image, cls_image = process_image(opencv_image)

    # Display the original image
    st.image(original_image, caption='Original Image', use_column_width=True)

    # Display the detected image
    if det_image is not None:
        st.image(det_image, caption='Detected Image', use_column_width=True)

    # Display the segmented image
    if seg_image is not None:
        st.image(seg_image, caption='Segmented Image', use_column_width=True)

    # Display the classified image
    if cls_image is not None:
        st.image(cls_image, caption='Classified Image', use_column_width=True)
