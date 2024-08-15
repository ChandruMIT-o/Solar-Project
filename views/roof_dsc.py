import PIL.IcnsImagePlugin
import streamlit as st
from ultralytics import YOLO
import PIL.Image
import cv2
import numpy as np
import os
from PIL import ImageFilter

def delete_files_in_directories(directories):
    count = 0
    for directory in directories:
        try:
            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    count += 1
                    # st.write(f"Deleted: {file_path}")
        except OSError as e:
            # st.error(f"Error deleting files in {directory}: {e}")
            pass
    
    st.write(f'{count} files got deleted.')

directories = ['output/cls', 'output/det', 'output/pseg', 'output/pseg_mask', 'output/seg', 'output/seg_mask']

if st.button("Delete Files"):
    delete_files_in_directories(directories)

def process_one_roof(det_cut_image, index, total, placement1, placement2):
    seg_results = models['Segmentation'](det_cut_image, verbose = False)
    placement1.code(f"Segmentation Completed : {index} / {total} roofs.")

    seg_path = f'output/seg/{image_name[:-3]}{index}.jpg'
    mask_path = f'output/seg_mask/{image_name[:-3]}{index}.jpg'

    masks = []
    combined_mask = []

    for res in seg_results[0].masks:
        image_height, image_width = seg_results[0].orig_shape

        mask = np.zeros((image_height, image_width), dtype=np.uint8)

        polygon_points = res.xy[0].astype(np.int32)

        cv2.fillPoly(mask, [polygon_points], 255)

        masks.append(mask)

    combined_mask = masks[0]

    for mask in masks[1:]:
        combined_mask = combined_mask + mask

    combined_mask = PIL.Image.fromarray(combined_mask)

    combined_mask = combined_mask.filter(ImageFilter.ModeFilter(size=15))

    cv2.imwrite(mask_path, combined_mask)
    
    image_array = np.array(det_cut_image)

    masked_image = np.zeros_like(image_array, dtype=np.uint8)
    for channel in range(3):
        masked_image[:,:,channel] = image_array[:,:,channel] * combined_mask

    det_cut_image_wb = PIL.Image.fromarray(masked_image)

    det_cut_image_wb.save(seg_path)

    part_seg_result = models['Part Segmentation'](det_cut_image_wb, verbose = False)
    placement2.code(f"Part Segmentation Completed : {index} / {total} roofs.")

    if part_seg_result is not None:

        part_seg_path = f'output/pseg/{image_name[:-3]}{index}.jpg'

        part_seg_result[0].save(part_seg_path)

        part_seg_image = PIL.Image.open(part_seg_path)

        index2 = 1

        if part_seg_result[0].masks is not None:

            for res in part_seg_result[0].masks:
                part_seg_mask_path = f'output/pseg_mask/{image_name[:-3]}{index}.{index2}.jpg'
                image_height, image_width = part_seg_result[0].orig_shape

                mask = np.zeros((image_height, image_width), dtype=np.uint8)

                polygon_points = res.xy[0].astype(np.int32)

                cv2.fillPoly(mask, [polygon_points], 255)

                mask = PIL.Image.fromarray(mask)
                smoothed_mask = mask.filter(ImageFilter.ModeFilter(size=10))

                smoothed_mask.save(part_seg_mask_path)

                index2 += 1


def sharpen_image(img):
  kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
  sharpened = cv2.filter2D(img, -1, kernel)
  return sharpened

def increase_contrast(img, contrast_factor = 1.5):

  img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  img[:, :, 2] = np.clip(img[:, :, 2] * contrast_factor, 0, 255)
  img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
  return img

@st.cache_data
def load_models():

    models = {
        "Detection" : YOLO("models/last_30e_det.pt"),
        "Segmentation" : YOLO("models/best_30e_seg.pt"),
        "Classification" : YOLO("models/best_30e_cls.pt"),
        "Part Segmentation" : YOLO("models/best_30e_part_seg.pt")
    }
    
    return models

models = load_models()

st.subheader("Roof Isolation")

col1, col2 = st.columns(2)

with col1:

    uploaded_files = st.file_uploader("Upload Images", accept_multiple_files=True)

    images = dict()
    for uploaded_file in uploaded_files:
        if uploaded_file is not None:
            images[uploaded_file.name] = PIL.Image.open(uploaded_file)

with col2:
    options = st.multiselect("Select Images:", options=images)

if options:
    for image_name in options:

        image = images[image_name]
        output_path = f"output/det/{image_name}"

        if not os.path.exists("output/det/"+image_name):

            with st.status("Processing...", expanded=True) as status:

                det_results = models['Detection'](np.array(image), verbose = False)
                for result in det_results:
                    boxes = result.boxes
                    result.save(filename=output_path)

                det_image = PIL.Image.open(output_path)
                    
                for result in det_results:

                    original_image = result.orig_img
                    boxes = result.boxes.xyxy

                    st.code(f'Buildings Detected: {len(boxes)}', language = None)
                    placement1 = st.empty()
                    placement2 = st.empty()

                    index = 1

                    for bb in boxes:
                        x, y, w, h = bb
                        x, y, w, h = int(x), int(y), int(w), int(h)

                        det_image = original_image[y:h , x:w]

                        det_cut_image = PIL.Image.fromarray(det_image)

                        det_path = f'output/det/{image_name[:-3]}{index}.jpg'

                        det_cut_image.save(det_path)

                        process_one_roof(det_cut_image, index, len(boxes), placement1, placement2)
                        
                        index += 1

                status.update(label="Process Completed!", state="complete", expanded=False)

        det_path = f'output/det/{image_name}'
        det_image = PIL.Image.open(det_path)

        det_cut_images = []

        index = 1
        det_cut_path = f'output/det/{image_name[:-3]}{index}.jpg'
        while os.path.exists(det_cut_path):
            det_cut_images.append(PIL.Image.open(det_cut_path))
            index += 1
            det_cut_path = f'output/det/{image_name[:-3]}{index}.jpg'

        seg_cut_images = []

        index = 1
        seg_cut_path = f'output/seg/{image_name[:-3]}{index}.jpg'
        while os.path.exists(seg_cut_path):
            seg_cut_images.append(PIL.Image.open(seg_cut_path))
            index += 1
            seg_cut_path = f'output/seg/{image_name[:-3]}{index}.jpg'
        
        pseg_cut_images = []

        index = 1
        pseg_cut_path = f'output/pseg/{image_name[:-3]}{index}.jpg'
        while os.path.exists(pseg_cut_path):
            pseg_cut_images.append(PIL.Image.open(pseg_cut_path))
            index += 1
            pseg_cut_path = f'output/pseg/{image_name[:-3]}{index}.jpg'

        with st.expander(image_name):
            c1, c2 = st.columns(2)
            with c1:
                st.image(image, caption = "Original image")
            with c2:
                st.image(det_image, caption = "Buildings Detected")

            i1, i2, i3 = st.columns(3)

            for a,b,c in zip(det_cut_images, seg_cut_images, pseg_cut_images):
                with i1:
                    st.image(a, caption = 'Detected Roof')
                with i2:
                    st.image(b, caption = 'Segmented Roof')
                with i3:
                    st.image(c, caption = 'Part Segmented Roof')
            

