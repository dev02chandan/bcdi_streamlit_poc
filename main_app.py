import os
import uuid
import io
from PIL import Image, ImageDraw, ImageFont
import base64
import json
import torch
from flask import Flask, render_template, request, redirect, jsonify
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
from collections import defaultdict

app = Flask(__name__)

# Load the YOLOv5 model
model = torch.hub.load("ultralytics/yolov5", "custom", path="best.pt", force_reload=True)
model.eval()


@app.route("/api/predict", methods=["POST"])
def predict():
    if not os.path.exists("static/bcdi_detection"):
        os.makedirs("static/bcdi_detection")
    if "file" not in request.files:
        return redirect(request.url)
    file = request.files["file"]

    if not file:
        return redirect(request.url)
    img_bytes = file.read()
    # originalImage = img_bytes
    img = Image.open(io.BytesIO(img_bytes))
    results = model([img])
    data = results.pandas().xyxy[0]
    df = pd.DataFrame(
        data, columns=["xmin", "ymin", "xmax", "ymax", "confidence", "class", "name"]
    )
    selected_data = df[["xmin", "ymin", "xmax", "ymax", "name"]].values.tolist()

    for entry in selected_data:
        def calculate_iou(box1, box2):
            # Calculate the intersection area
            x_min = max(box1[0], box2[0])
            y_min = max(box1[1], box2[1])
            x_max = min(box1[2], box2[2])
            y_max = min(box1[3], box2[3])

            intersection_area = max(0, x_max - x_min + 1) * max(0, y_max - y_min + 1)

            # Calculate the union area
            box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
            box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

            union_area = box1_area + box2_area - intersection_area
            iou = intersection_area / union_area

            return iou

    # non_max_suppression function
    def non_max_suppression(selected_data, iou_threshold):
        sorted_boxes = sorted(selected_data, key=lambda box: box[4], reverse=True)
        selected_boxes = []

        for box in sorted_boxes:
            if not any(
                    calculate_iou(box, selected_box) > iou_threshold
                    for selected_box in selected_boxes
            ):
                selected_boxes.append(box)

        return selected_boxes

    iou_threshold = 0.8
    selected_bounding_boxes = non_max_suppression(selected_data, iou_threshold)

    # Convert the list of lists into a DataFrame
    columns = ["xmin", "ymin", "xmax", "ymax", "class"]
    df = pd.DataFrame(selected_bounding_boxes, columns=columns)
    df = df.sort_values(by=["ymin"], ascending=True)
    df = df.reset_index(drop=True)

    draw = ImageDraw.Draw(img)
    font_size = 12  # Adjust the font size as needed
    font = ImageFont.truetype("ProximaNova-Regular.otf", font_size)
    sorted_data = sorted(selected_bounding_boxes, key=lambda x: x[1])
    print(sorted_data)

    response_list = []
    for index, annotation in enumerate(sorted_data, start=1):
        entry = {"index": index, "class": annotation[4], "BBox": tuple(annotation[0:4])}
        response_list.append(entry)
        x_min, y_min, x_max, y_max, class_name = annotation
        draw.rectangle([x_min, y_min, x_max, y_max], outline="yellow", width=1)
        draw.text((x_min, y_min), f"{index}: {class_name}", fill="white", font=font)

    image_name = f"{str(uuid.uuid4())}.png"
    image_path = os.path.join("static/bcdi_detection", image_name)
    img.save(image_path)
    public_image_url = f"{request.url_root}static/bcdi_detection/{image_name}"

    ######################## class percentage ###################################

    class_counts = {}

    for entry in sorted_data:
        class_label = entry[-1]
        if class_label in class_counts:
            class_counts[class_label] += 1
        else:
            class_counts[class_label] = 1

    total_entries = len(sorted_data)
    class_percentage = {}

    for class_label, count in class_counts.items():
        percentage = (count / total_entries) * 100
        percentage = round(percentage, 2)
        class_percentage[class_label] = percentage

    ########## image size ###################
    max_x = 0
    max_y = 0

    for entry in sorted_data:
        xmax = entry[2]
        ymax = entry[3]

        if xmax > max_x:
            max_x = xmax
        if ymax > max_y:
            max_y = ymax

    image_size = (int(max_x), int(max_y))

    response = {
        "Class": total_entries,
        "Percentage": class_percentage,
        "image": public_image_url,
        "image size": image_size,
        "index_dt": response_list,
    }

    return jsonify({"data": response})


@app.route("/api/update_class", methods=["POST"])
def update_class():
    index_dt = request.form.get("index_dt")
    image_file = request.files.get("image")
    if index_dt is None or image_file is None:
        return jsonify({"error": "Missing data in the request"}), 400

    try:
        index_dt = json.loads(index_dt)
        if not isinstance(index_dt, list):
            raise ValueError("index_dt should be a list of annotations")
    except (json.JSONDecodeError, ValueError) as e:
        return jsonify({"error": f"Invalid index_dt format {str(e)}"}), 400

    class_counts = defaultdict(int)
    total_annotations = len(index_dt)

    annotations = []
    # img = Image.open(io.BytesIO(image_file))
    img = Image.open(image_file)
    draw = ImageDraw.Draw(img)
    font_size = 15  # Adjust the font size as needed
    font = ImageFont.truetype("ProximaNova-Regular.otf", font_size)

    for item in index_dt:
        index = item["index"]
        class_name = item["class"]
        bbox_values = item["BBox"]
        x_min, y_min, x_max, y_max = bbox_values
        annotation = {
            "index": index,
            "class": class_name,
            "BBox": [x_min, y_min, x_max, y_max],
        }
        annotations.append(annotation)

        class_counts[class_name] += 1

        # Draw the bounding box
        draw.rectangle([x_min, y_min, x_max, y_max], outline="green", width=1)

        # Draw the class label
        text = f"{index}: {class_name}"
        text_bbox = draw.textbbox((x_min, y_min - font_size), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        draw.rectangle([x_min, y_min, x_max, y_max], outline="green", width=1)
        draw.rectangle(
            [x_min, y_min - text_height, x_min + text_width, y_min],
            fill="green",
        )
        draw.text((x_min, y_min - text_height), text, font=font, fill="white")

    image_name = f"{str(uuid.uuid4())}.png"
    image_path = os.path.join("static/bcdi_detection", image_name)
    img.save(image_path)
    image_url = f"{request.url_root}static/bcdi_detection/{image_name}"

    class_percentages = {
        class_name: round(count / total_annotations * 100, 2)
        for class_name, count in class_counts.items()
    }

    response = {"index_dt": annotations, "Percentage": class_percentages, "image": image_url}

    return jsonify({"data": response})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
