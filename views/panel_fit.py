import streamlit as st
import os
from PIL import Image
import re
import numpy as np
import cv2
from tqdm import tqdm
import json
import pandas as pd

def remove_zero_rows_cols(arr):
  rows_to_keep = np.any(arr != 0, axis=1)
  cols_to_keep = np.any(arr != 0, axis=0)

  return arr[rows_to_keep][:, cols_to_keep]

def create_panel(height, width, angle):

    panel_size = (max(height, width) * 2, max(height, width) * 2)

    panel = np.zeros(panel_size, dtype=np.uint8)

    center = (panel_size[1] // 2, panel_size[0] // 2)

    rect = np.array([
        [-width // 2, -height // 2],
        [width // 2, -height // 2],
        [width // 2, height // 2],
        [-width // 2, height // 2]
    ])

    rotation_matrix = cv2.getRotationMatrix2D(center=(0, 0), angle=angle, scale=1.0)

    rotated_rect = np.dot(rect, rotation_matrix[:, :2].T).astype(int)

    translated_rect = rotated_rect + np.array(center)

    cv2.fillConvexPoly(panel, translated_rect, 1)

    return remove_zero_rows_cols(panel)

def count_ones(array): return (array == 1).sum()

def remove_black_rows_cols(img):

  black_threshold = 0

  rows_to_keep = np.any(img > black_threshold, axis=(1, 2))
  img = img[rows_to_keep]

  cols_to_keep = np.any(img > black_threshold, axis=(0, 2))
  img = img[:, cols_to_keep]

  return img

def convert_to_binary(color_image):

    binary_image = np.all(color_image != 0, axis=-1).astype(np.uint8)

    return binary_image

def create_color_panel(height, width, angle):

    height, width = height - 2, width - 2

    panel_size = (max(height, width) * 2, max(height, width) * 2)

    panel = np.zeros((panel_size[0], panel_size[1], 3), dtype=np.uint8)

    center = (panel_size[1] // 2, panel_size[0] // 2)

    rect = np.array([
        [-width // 2, -height // 2],
        [width // 2, -height // 2],
        [width // 2, height // 2],
        [-width // 2, height // 2]
    ])

    rotation_matrix = cv2.getRotationMatrix2D(center=(0, 0), angle=angle, scale=1.0)

    rotated_rect = np.dot(rect, rotation_matrix[:, :2].T).astype(int)

    translated_rect = rotated_rect + np.array(center)

    cv2.fillConvexPoly(panel, translated_rect, (4, 1, 84))

    cv2.polylines(panel, [translated_rect], isClosed=True, color=(0, 255, 255), thickness=2)

    return remove_black_rows_cols(panel)

def place_panel(image1, panel, x, y, w, h):

    roi = image1[y:y+h, x:x+w]

    mask = np.all(panel != [0, 0, 0], axis=-1)

    roi[mask] = panel[mask]

    image1[y:y+h, x:x+w] = roi

    return image1

def panel_placement(matrix, height, width, angle):   

    heig, widt = matrix.shape

    panel = convert_to_binary(create_color_panel(height, width, angle))
    inv_panel = 1 - panel

    indx = 0
    indy = 0
    height , width = panel.shape
    size = count_ones(panel)

    placements = []

    indx = 0
    while indx < heig - height - 1:
        indy = 0
        skip = 0
        while indy < widt - width - 1:
            
            sample = matrix[indx:indx+height, indy:indy+width]
            tsize = count_ones(sample * panel)
            
            if tsize == size:
                skip = indx
                placements.append([indx, indy])
                mask = inv_panel == 0
                matrix[indx:indx+height, indy:indy+width][mask] = inv_panel[mask]
            
            indy += 1
        if skip == 0:
            indx += 1
        else:
            indx = skip

    return (matrix, placements)

def find_placements(matrix, panel_height = 10, panel_weight = 20, my_bar = st.empty()):
    recovery_matrix = matrix.copy()

    panel_count = dict()

    for angle in tqdm(range(0, 180, 5)):

        matrix, placements = panel_placement(matrix, panel_height, panel_weight, angle)

        panel_count[angle] = len(placements)

        matrix = recovery_matrix.copy()

        my_bar.progress(int((angle / 180) * 100), text=f"Calculating optimal angle {int((angle / 180) * 100)}%...")

    optimal_angle = max(panel_count, key=panel_count.get)
    # matrix, placements = panel_placement(matrix, panel_height, panel_weight, optimal_angle)

    return optimal_angle

def list_files_in_directory(directory):

    file_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            if os.path.isfile(file_path):
                file_paths.append(file_path[11:])
    return file_paths

PIXEL_RESOLUTION = 0.075
PANEL_HEIGHT = 20
PANEL_WIDTH = 15
OPTIMAL_ANGLE = 0
FIND_OPTIMAL_ANGLE = True

with st.expander('Solar Panel Parameters: '):
    col2, col3 = st.columns(2)

    with col2:
        PIXEL_RESOLUTION = float(st.text_input("Spatial Resolution of Imagery (m)", placeholder=PIXEL_RESOLUTION, value=PIXEL_RESOLUTION))
        OPTIMAL_ANGLE = int(st.text_input("Optimum Tilt Angle (degrees):", placeholder=OPTIMAL_ANGLE, value=OPTIMAL_ANGLE))
    with col3:
        PANEL_HEIGHT = int(st.text_input("Panel Height (m): ", placeholder= PANEL_HEIGHT, value=PANEL_HEIGHT))
        PANEL_WIDTH = int(st.text_input("Panel Width (m): ", placeholder= PANEL_WIDTH, value=PANEL_WIDTH))

    FIND_OPTIMAL_ANGLE = st.checkbox('Auto Detect Optimum Angle for Maximised Panel Packing', value=True)

    st.image(create_color_panel(PANEL_HEIGHT, PANEL_WIDTH, OPTIMAL_ANGLE))

image_ids = list(set([m.split('.')[0] for m in list_files_in_directory('output\\det')]))

image_id = st.selectbox("Select image name:", image_ids)

if image_id:

    det_cut_paths = []
    det_cut_images = []

    index = 1
    det_cut_path = f'output/det/{image_id}.{index}.jpg'
    while os.path.exists(det_cut_path):
        det_cut_paths.append(f'{image_id}.{index}.jpg')
        det_cut_images.append(Image.open(det_cut_path))
        index += 1
        det_cut_path = f'output/det/{image_id}.{index}.jpg'

    selected_image = None
    
    with st.expander(f"Selected Building:"):
        cols = st.columns(8)

        for idx, img in enumerate(det_cut_images):
            with cols[idx % 8]:
                st.image(img, caption=os.path.basename(det_cut_paths[idx]))

        instance_id = image_id = st.selectbox("Select Roof:", det_cut_paths)

    if instance_id:
        st.write(instance_id)

        cl1, cl2 = st.columns(2)

        selected_image = Image.open(f'output/det/{instance_id}')

        with cl1:
            st.image(selected_image)

        pseg_cut_names = []
        pseg_cut_paths = []
        pseg_cut_images = []

        index2 = 1
        pseg_cut_path = f'output/pseg_mask/{instance_id[:-4]}.{index2}.jpg'
        while os.path.exists(pseg_cut_path):
            pseg_cut_paths.append(pseg_cut_path)
            pseg_cut_names.append(f'{instance_id[:-4]}.{index2}.jpg')
            pseg_cut_images.append(Image.open(pseg_cut_path))
            index2 += 1
            pseg_cut_path = f'output/pseg_mask/{instance_id[:-4]}.{index2}.jpg'
        
        temp_image = np.array(selected_image).copy()

        file_path = "output/data.json"

        if not os.path.exists(file_path):
            with open(file_path, "w") as file:
                json.dump({}, file)        

        TOTAL_ROOF_AREA = 0
        TOTAL_PANELS = 0    

        for seg_mask, seg_mask_name in zip(pseg_cut_paths, pseg_cut_names):
            with open(file_path, "r") as file:
                data = json.load(file)
            
            mask = np.array(Image.open(seg_mask)) / 255
            mask = np.where(mask > 1, 1, mask)

            EFFECTIVE_AREA = count_ones(mask) * PIXEL_RESOLUTION

            TOTAL_ROOF_AREA += EFFECTIVE_AREA

            if FIND_OPTIMAL_ANGLE:

                if seg_mask_name not in data:

                    my_bar = st.progress(0, text="Calculating optimal angle.")

                    DETECTED_OPTIMAL_ANGLE = find_placements(mask, PANEL_HEIGHT, PANEL_WIDTH, my_bar)

                    my_bar.progress(100, text=f"Calculated 100%")
                    data[seg_mask_name] = DETECTED_OPTIMAL_ANGLE
                    with open(file_path, "w") as file:
                        json.dump(data, file)

            if FIND_OPTIMAL_ANGLE:
                angle = int(data[seg_mask_name])
                _, panel_placements = panel_placement(mask, PANEL_HEIGHT, PANEL_WIDTH, int(data[seg_mask_name]))
            else:
                angle = int(OPTIMAL_ANGLE)
                _, panel_placements = panel_placement(mask, PANEL_HEIGHT, PANEL_WIDTH, int(OPTIMAL_ANGLE))

            panel = create_color_panel(PANEL_HEIGHT, PANEL_WIDTH, angle)
            
            colm1, colm2, colm3, colm4, colm5 = st.columns([1, 1, 5, 5, 5])

            with colm1:
                st.image(panel)
            with colm2:
                st.image(mask)
            with colm3:
                st.code(f'Optimum angle: {angle}°')
            with colm4:
                st.code(f'Maximum fit: {len(panel_placements)} panels')
                TOTAL_PANELS += len(panel_placements)
            with colm5:
                st.code(f'Effective Area: {round(EFFECTIVE_AREA, 2)} m²')

            wp, hp, _ = panel.shape

            for xp, yp in panel_placements:
                temp_image = place_panel(temp_image, panel, yp, xp, hp, wp)

        with cl2:
            st.image(temp_image)

        st.code(f'Total Effective Area: {TOTAL_ROOF_AREA} m²')
        st.code(f'Total Number of Panels: {TOTAL_PANELS}')
        