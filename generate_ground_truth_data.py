import cv2
import numpy as np
import json
import os

def load_points_from_file(image_path, is_pred, pred_file=None):
    if is_pred:
        # Use the provided pred_file path for predictions
        point_cloud_path = pred_file
    else:
        # Extract index from the image path and load ground truth
        ind = image_path.split('_')[-1].split('.')[0]
        point_cloud_path = f"./cmr_base_centroids/instance-{ind}.txt"
    
    try:
        with open(point_cloud_path, 'r') as file:
            points = json.load(file)
        return points
    except FileNotFoundError:
        print(f"No points file found: {point_cloud_path}")
        return []

def display_combined_annotations(image_path, pred_file, output_folder):
    # Extract index from the prediction file name
    ind = pred_file.split('_')[-1].split('.')[0]
    print(f"Annotating image #{ind}")

    # Read the image
    image = cv2.imread(image_path).copy()

    if image is None:
        print(f"Image {image_path} not found or couldn't be loaded.")
        return

    height, width = image.shape[:2]
    square_size = 7  # Half the size of the square around points

    # Load true points
    true_points = load_points_from_file(image_path, is_pred=False)

    # Load predicted points
    pred_points = load_points_from_file(image_path, is_pred=True, pred_file=pred_file)

    # Annotate true points in green
    for point in true_points:
        conceptual_x, conceptual_y, _ = point
        pixel_x = int((1 - (conceptual_x / 40)) * width)
        pixel_y = int((1 - (conceptual_y + 20) / 40) * height)

        if 0 <= pixel_x < width and 0 <= pixel_y < height:
            # Draw green circle and black square
            cv2.circle(image, (pixel_y, pixel_x), radius=2, color=(0, 255, 0), thickness=-1)
            top_left = (pixel_y - square_size, pixel_x - square_size)
            bottom_right = (pixel_y + square_size, pixel_x + square_size)
            cv2.rectangle(image, top_left, bottom_right, color=(0, 255, 0), thickness=2)
        else:
            print(f"True Point (x: {pixel_x}, y: {pixel_y}) is out of bounds for the image.")

    # Annotate predicted points in blue
    for point in pred_points:
        conceptual_x, conceptual_y, _ = point
        pixel_x = int((1 - (conceptual_x / 40)) * width)
        pixel_y = int((1 - (conceptual_y + 20) / 40) * height)

        if 0 <= pixel_x < width and 0 <= pixel_y < height:
            # Draw blue circle and black square
            cv2.circle(image, (pixel_y, pixel_x), radius=2, color=(255, 0, 0), thickness=-1)
            top_left = (pixel_y - square_size, pixel_x - square_size)
            bottom_right = (pixel_y + square_size, pixel_x + square_size)
            cv2.rectangle(image, top_left, bottom_right, color=(255, 0, 0), thickness=2)
        else:
            print(f"Pred Point (x: {pixel_x}, y: {pixel_y}) is out of bounds for the image.")

    # Save the combined annotated image in the appropriate folder
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, f"annotated_{ind}.png")
    cv2.imwrite(output_path, image)

# Identify prediction files and process them
pred_files = [f for f in os.listdir('./preds') if f.endswith('.txt')]
train_files = [f for f in pred_files if 'train' in f]
test_files = [f for f in pred_files if 'test' in f]

# Annotate training predictions
for train_file in train_files:
    ind = train_file.split('_')[-1].split('.')[0]
    image_path = f'./cropped_images/point_cloud_visualization_{ind}.png'
    display_combined_annotations(image_path, f'./preds/{train_file}', './vis_preds_train')

# Annotate testing predictions
for test_file in test_files:
    ind = test_file.split('_')[-1].split('.')[0]
    image_path = f'./cropped_images/point_cloud_visualization_{ind}.png'
    display_combined_annotations(image_path, f'./preds/{test_file}', './vis_preds_test')