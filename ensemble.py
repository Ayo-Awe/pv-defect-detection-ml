import json
import torch
import numpy as np
from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion, nms
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Load YOLOv5 model
yolov5 = torch.hub.load('ultralytics/yolov5', 'custom',
                        path="./yolov5/runs/train/yolov5n_results/weights/best.pt")
yolov8 = YOLO("./runs/detect/train2/weights/best.pt")


# Load custom dataset
dataset_path = './datasets/valid/_annotations.coco.json'
coco = COCO(dataset_path)


# Function to run inference
def run_inference(model, image_ids, img_dir):
    results = []
    for img_id in image_ids:
        img_info = coco.loadImgs(img_id)[0]
        img_path = f"{img_dir}/{img_info['file_name']}"

        preds = None
        boxes = None
        scores = None
        labels = None

        try:
            preds = model(img_path)[0].boxes
            boxes = preds.xyxy.cpu().numpy()  # Bounding boxes
            scores = preds.conf.cpu().numpy()  # Confidence scores
            labels = preds.cls.cpu().numpy()  # Class labels
        except:
            preds = model(img_path)
            boxes = preds.xyxy[0][:, :4].cpu().numpy()  # Bounding boxes
            scores = preds.xyxy[0][:, 4].cpu().numpy()  # Confidence scores
            labels = preds.xyxy[0][:, 5].cpu().numpy()  # Class labels

        # Extract boxes, scores, and labels

        results.append({
            'image_id': img_id,
            'boxes': boxes,
            'scores': scores,
            'labels': labels
        })
    return results


def convert_to_coco_format(img_ids, boxes, scores, labels):
    results = []
    for i, img_id in enumerate(img_ids):
        for box, score, label in zip(boxes[i], scores[i], labels[i]):
            result = {
                'image_id': img_id,
                'category_id': int(label),
                # Convert to COCO format (x, y, width, height)
                'bbox': [box[0], box[1], box[2]-box[0], box[3]-box[1]],
                'score': float(score)
            }
            results.append(result)
    return results


# Get image ids
img_ids = coco.getImgIds()
image_dir = "./datasets/valid/"

# Run inference on custom dataset
print("running yolov8 predictions")
yolov8_preds = run_inference(yolov8, img_ids, image_dir)
print("finished yolov8 predictions")
# Save results to a JSON file
with open('v8_predictoins.json', 'w') as f:
    json.dump(yolov8_preds, f)


print("running yolov5 predictions")
yolov5_preds = run_inference(yolov5, img_ids, image_dir)
print("finished yolov5 predictions")
with open('v5_predictoins.json', 'w') as f:
    json.dump(yolov5_preds, f)


# Function to ensemble predictions


def ensemble_predictions(preds1, preds2, iou_thr=0.5):
    boxes_list = []
    scores_list = []
    labels_list = []

    for i in range(len(preds1)):
        boxes1, scores1, labels1 = preds1[i]['boxes'], preds1[i]['scores'], preds1[i]['labels']
        boxes2, scores2, labels2 = preds2[i]['boxes'], preds2[i]['scores'], preds2[i]['labels']

        boxes = np.vstack((boxes1, boxes2))
        scores = np.hstack((scores1, scores2))
        labels = np.hstack((labels1, labels2))

        print(boxes, scores, lables)

        boxes, scores, labels = nms(
            [boxes1, boxes2], [scores1, scores2], [labels1, labels2], weights=[2, 1], iou_thr=iou_thr)

        boxes_list.append(boxes)
        scores_list.append(scores)
        labels_list.append(labels)

    return boxes_list, scores_list, labels_list


# Ensemble the predictions
ensemble_boxes, ensemble_scores, ensemble_labels = ensemble_predictions(
    yolov5_preds, yolov8_preds)

# Convert predictions to COCO format


coco_results = convert_to_coco_format(
    img_ids, ensemble_boxes, ensemble_scores, ensemble_labels)

# Save results to a JSON file
with open('ensemble_results.json', 'w') as f:
    json.dump(coco_results, f)

# Evaluate using COCOEvaluator
coco_dt = coco.loadRes('ensemble_results.json')
coco_eval = COCOeval(coco, coco_dt, 'bbox')
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()
