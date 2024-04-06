import torch
import supervision
import transformers
import pytorch_lightning
import timm
from transformers import DetrImageProcessor
import cv2
import matplotlib.pyplot as plt
from transformers import DetrForObjectDetection
MODEL_PATH = 'detr_model_100'

# loading model
model = DetrForObjectDetection.from_pretrained(MODEL_PATH)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(DEVICE)



id2label = { 0: 'bird', 1:'boar', 2:'dog', 3:'dragon', 4:'horse', 5:'monkey', 6:'ox',
             7:'rabbit', 8:'rat', 9:'sheep', 10:'snake', 11:'tiger'}

image_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
box_annotator = supervision.BoxAnnotator()

# Define the video file path or camera index  # Change this to your video file path
# Or use a camera stream by providing the camera index (0 for the default camera)
# video_path = 0

# Create a VideoCapture object to read the video
cap = cv2.VideoCapture(0)

# Check if the video opened successfully
if not cap.isOpened():
    print("Error: Unable to open video.")
    exit()

# Loop through the video frames
while True:
    # Read a frame from the video
    ret, frame = cap.read()

    # Check if the frame was read successfully
    if not ret:
        print("Error: Unable to read frame.")
        break

    # Perform object detection on the frame
    with torch.no_grad():
        # Load image and predict
        inputs = image_processor(images=frame, return_tensors='pt').to(DEVICE)
        outputs = model(**inputs)

        # Post-process
        target_sizes = torch.tensor([frame.shape[:2]]).to(DEVICE)
        results = image_processor.post_process_object_detection(
            outputs=outputs,
            threshold=0.5,
            target_sizes=target_sizes
        )[0]

        # Extract detections
        detections = supervision.Detections.from_transformers(transformers_results=results)
        #print(detections)
        labels = [f"{id2label[class_id]} {confidence:.2f}" for _, confidence, class_id, _ in detections]
        frame_detections = box_annotator.annotate(scene=frame.copy(), detections=detections, labels=labels)
        print(id2label[detections.class_id])

    # Display the frame with detections
    cv2.imshow('frame', frame_detections)

    # Check for the 'q' key to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
