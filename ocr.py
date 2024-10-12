import os
import time
import math
import json
import cv2
from PIL import Image, ImageEnhance
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from msrest.authentication import CognitiveServicesCredentials

# Table Formatter
def convert_to_table_format(items):
    """
    Convert a list of items to a table format string.
    Each item is separated by ' | ' except the last one, which is followed by ' &\n'.
    """
    return ' | '.join(map(str, items[:-1])) + ' ' + str(items[-1])

# Geometry Functions
def distance(x1, y1, x2, y2):
    """Calculate the Euclidean distance between two points."""
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def angle_between_points(x1, y1, x2, y2):
    """Calculate the angle between two points with respect to the origin."""
    return math.atan2(y2 - y1, x2 - x1)

def sector_area_ratio(points):
    """
    Calculate the area ratio of a sector based on three points (center, edge1, edge2).
    The ratio is the sector area divided by the total circle area.
    """
    cx, cy, ex1, ey1, ex2, ey2 = points
    radius = distance(cx, cy, ex1, ey1)
    theta = abs(angle_between_points(cx, cy, ex2, ey2) - angle_between_points(cx, cy, ex1, ey1))
    theta = theta if theta <= 2 * math.pi else 2 * math.pi - theta
    sector_area = 0.5 * theta * radius ** 2
    circle_area = math.pi * radius ** 2
    return sector_area / circle_area

def sector_area(points):
    """Calculate the area of a sector given three critical points."""
    cx, cy, ex1, ey1, ex2, ey2 = points
    radius = distance(cx, cy, ex1, ey1)
    theta = abs(angle_between_points(cx, cy, ex2, ey2) - angle_between_points(cx, cy, ex1, ey1))
    theta = theta if theta <= 2 * math.pi else 2 * math.pi - theta
    return 0.5 * theta * radius ** 2

# OCR and Data Processing
def find_closest_text(input_bbox, OCR_results):
    """
    Find the closest OCR text to the input bounding box.
    """
    if not OCR_results:
        return None
    
    def dist(bbox1, bbox2):
        return ((bbox1[0] - bbox2[0]) ** 2 + (bbox1[1] - bbox2[1]) ** 2) ** 0.5

    closest = min(OCR_results, key=lambda r: dist(input_bbox, r['bbox']))
    OCR_results.remove(closest)
    return closest['text']

def is_decimal(value):
    """Check if a string is a decimal number."""
    try:
        float(value)
        return True
    except ValueError:
        return False

def data_range_estimation(Right, Bottom, OCR_results):
    """
    Estimate the data range (rmin, rmax) from OCR results within a boundary.
    """
    candidates = [r for r in OCR_results if r['bbox'][0] + r['bbox'][2] < Right - 4 and is_decimal(r['text'])]
    filtered_candidates = [r for r in candidates if float(r['text']) != 0]

    rmin = min(candidates, key=lambda r: abs(r['bbox'][0]) + abs(r['bbox'][1] - Bottom))
    rmax = min(filtered_candidates, key=lambda r: abs(r['bbox'][0]) + abs(r['bbox'][1]))

    return {'num': float(rmax['text']), 'bbox': rmax['bbox']}, {'num': float(rmin['text']), 'bbox': rmin['bbox']}

def calculate_bar_val(max_val, max_val_bbox, min_val, min_val_bbox, val_bbox):
    """
    Calculate the bar value based on the bounding box positions of max, min, and val.
    """
    y_scale = (max_val - min_val) / (min_val_bbox[1] - max_val_bbox[1])
    return min_val + (min_val_bbox[1] - val_bbox[1]) * y_scale

def filter_no_number(OCR_results):
    """
    Filter out OCR results that contain numbers.
    """
    return [r for r in OCR_results if not any(char.isdigit() for char in r['text'])]

def delete_y_label(min_bbox, texts):
    """
    Remove y-labels from text based on their bounding box proximity.
    """
    x_min, y_min, w, h = min_bbox
    top_right_x, bottom_right_y = x_min + w, y_min + h

    return [t for t in texts if not (abs(top_right_x - (t['bbox'][0] + t['bbox'][2])) <= 5 
                                     and abs(bottom_right_y - (t['bbox'][1] + t['bbox'][3])) <= 5)]

# Bounding Box Utilities
def convert_to_xywh(bbox):
    """Convert bounding box to (x, y, w, h) format."""
    x_coords, y_coords = bbox[0::2], bbox[1::2]
    return min(x_coords), min(y_coords), max(x_coords) - min(x_coords), max(y_coords) - min(y_coords)

def find_topmost_bbox(ocr_results):
    """Find the topmost bounding box based on y-coordinates."""
    return min(ocr_results, key=lambda r: r['bbox'][1])

def find_bottommost_bbox(ocr_results):
    """Find the bottommost bounding box based on y-coordinates."""
    return max(ocr_results, key=lambda r: r['bbox'][1] + r['bbox'][3])

def find_leftmost_bbox(ocr_results):
    """Find the leftmost bounding box based on x-coordinates."""
    return min(ocr_results, key=lambda r: r['bbox'][0])

# OCR with Azure
def ocr_with_azure():
    """
    Perform OCR using Azure Cognitive Services and return the results in Tesseract format.
    """
    subscription_key = os.getenv("VISION_KEY")
    endpoint = os.getenv("VISION_ENDPOINT")
    client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(subscription_key))

    # Enhance image contrast
    img = Image.open("./test_img.png")
    img = ImageEnhance.Contrast(img).enhance(2.0)
    img.save('OCR_temp.png')

    with open("OCR_temp.png", "rb") as image:
        read_response = client.read_in_stream(image, raw=True)

    # Get operation ID and poll the status
    operation_id = read_response.headers["Operation-Location"].split("/")[-1]
    while True:
        result = client.get_read_result(operation_id)
        if result.status not in ['notStarted', 'running']:
            break
        time.sleep(1)

    tesseract_format = {'left': [], 'top': [], 'width': [], 'height': [], 'text': []}
    if result.status == OperationStatusCodes.succeeded:
        for text_result in result.analyze_result.read_results:
            for line in text_result.lines:
                tesseract_format['text'].append(line.text)
                x, y, w, h = convert_to_xywh(line.bounding_box)
                tesseract_format['left'].append(x)
                tesseract_format['top'].append(y)
                tesseract_format['width'].append(w)
                tesseract_format['height'].append(h)
    
    return tesseract_format

# OCR Function for Charts
def Ocr(chart_type):
    """
    Perform OCR on a chart and calculate data ranges based on bounding box values.
    """
    image = cv2.imread("./test_img.png")
    height, width, _ = image.shape
    json_result = {"...": "results"}  # Add logic for processing chart types here.
    return json_result
