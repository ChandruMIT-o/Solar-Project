import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm

def png_to_matrix(image_path, threshold=127):

  img = Image.open(image_path).convert('L') 

  img_array = np.array(img)

  binary_image = (img_array > threshold).astype(np.uint8)

  return np.array(binary_image)

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

def panel_placement(matrix, height, width, angle):   

    heig, widt = matrix.shape

    panel = create_panel(height, width, angle)
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

def find_placements(matrix, panel_height = 10, panel_weight = 20):
    recovery_matrix = matrix.copy()

    panel_count = dict()

    for angle in tqdm(range(0, 180, 5)):

        matrix, placements = panel_placement(matrix, panel_height, panel_weight, angle)

        panel_count[angle] = len(placements)

        matrix = recovery_matrix.copy()

    optimal_angle = max(panel_count, key=panel_count.get)
    matrix, placements = panel_placement(matrix, panel_height, panel_weight, optimal_angle)

    return optimal_angle, placements

def remove_black_rows_cols(img):

  black_threshold = 0

  rows_to_keep = np.any(img > black_threshold, axis=(1, 2))
  img = img[rows_to_keep]

  cols_to_keep = np.any(img > black_threshold, axis=(0, 2))
  img = img[:, cols_to_keep]

  return img

def create_color_panel(height, width, angle):

    height, width = height - 1, width - 1

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

    cv2.polylines(panel, [translated_rect], isClosed=True, color=(255, 255, 255), thickness=1)

    return remove_black_rows_cols(panel)

def place_panel(image, panel, x, y, w, h):

    roi = image[y:y+h, x:x+w]

    mask = np.all(panel != [0, 0, 0], axis=-1)

    roi[mask] = panel[mask]

    image[y:y+h, x:x+w] = roi

    return image
