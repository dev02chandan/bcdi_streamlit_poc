import streamlit as st
import torch
from PIL import Image, ImageDraw, ImageFont
import io
import os
import uuid
import pandas as pd
from collections import defaultdict
import json

# Load the YOLOv5 model
@st.cache_resource
def load_model():
    model = torch.hub.load("ultralytics/yolov5", "custom", path="best.pt")
    model.eval()
    return model

model = load_model()

def calculate_iou(box1, box2):
    x_min = max(box1[0], box2[0])
    y_min = max(box1[1], box2[1])
    x_max = min(box1[2], box2[2])
    y_max = min(box1[3], box2[3])

    intersection_area = max(0, x_max - x_min + 1) * max(0, y_max - y_min + 1)

    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    union_area = box1_area + box2_area - intersection_area
    iou = intersection_area / union_area

    return iou

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

def predict(img):
    results = model([img])
    data = results.pandas().xyxy[0]
    df = pd.DataFrame(
        data, columns=["xmin", "ymin", "xmax", "ymax", "confidence", "class", "name"]
    )
    selected_data = df[["xmin", "ymin", "xmax", "ymax", "name"]].values.tolist()

    iou_threshold = 0.8
    selected_bounding_boxes = non_max_suppression(selected_data, iou_threshold)

    return selected_bounding_boxes

def draw_boxes(img, sorted_data):
    draw = ImageDraw.Draw(img)
    font_size = 60
    font = ImageFont.truetype("ProximaNova-Regular.otf", font_size)

    response_list = []
    for index, annotation in enumerate(sorted_data, start=1):
        entry = {"index": index, "class": annotation[4], "BBox": tuple(annotation[0:4])}
        response_list.append(entry)
        x_min, y_min, x_max, y_max, class_name = annotation
        draw.rectangle([x_min, y_min, x_max, y_max], outline="yellow", width=5)
        draw.text((x_min, y_min), f"{index}: {class_name}", fill="red", font=font)

    return img, response_list

def main():
    st.title("Maitri Diamond Impurity Detection")

    uploaded_file = st.file_uploader("Choose an image...", type="png")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Detect Objects"):
            sorted_data = predict(image)
            annotated_img, response_list = draw_boxes(image.copy(), sorted_data)

            st.image(annotated_img, caption="Annotated Image", use_column_width=True)

            # Calculate class percentages
            class_counts = defaultdict(int)
            total_entries = len(sorted_data)
            for entry in sorted_data:
                class_counts[entry[4]] += 1

            class_percentages = {
                class_name: round(count / total_entries * 100, 2)
                for class_name, count in class_counts.items()
            }

            st.write("Class Percentages:")
            st.json(class_percentages)

            st.write("Detected Objects:")
            st.json(response_list)

            # Allow user to update classes
            st.subheader("Update Classes")
            updated_data = st.text_area("Edit the JSON data below to update classes:", json.dumps(response_list, indent=2))

            if st.button("Update"):
                try:
                    updated_data = json.loads(updated_data)
                    updated_img = image.copy()
                    updated_img, _ = draw_boxes(updated_img, [(item['BBox'][0], item['BBox'][1], item['BBox'][2], item['BBox'][3], item['class']) for item in updated_data])
                    st.image(updated_img, caption="Updated Image", use_column_width=True)

                    # Recalculate class percentages
                    updated_class_counts = defaultdict(int)
                    for item in updated_data:
                        updated_class_counts[item['class']] += 1

                    updated_class_percentages = {
                        class_name: round(count / len(updated_data) * 100, 2)
                        for class_name, count in updated_class_counts.items()
                    }

                    st.write("Updated Class Percentages:")
                    st.json(updated_class_percentages)

                except json.JSONDecodeError:
                    st.error("Invalid JSON format. Please check your input.")


if __name__ == "__main__":
    main()