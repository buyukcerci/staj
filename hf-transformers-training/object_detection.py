
from transformers import pipeline

from PIL import Image
import cv2

IMAGE_NAME = "mask.jpg"

# Initialize the object detection pipeline with a pre-trained model
detector = pipeline(task="object-detection", model="facebook/detr-resnet-50")

# Load an image for object detection
image = Image.open(IMAGE_NAME)
print(type(image))

# Load an image for drawing bounding boxes using OpenCV
image_cv = cv2.imread(IMAGE_NAME)

# Perform object detection on the image
results = detector(image)

boxes = []

# Process the results and extract bounding boxes, labels, and scores 
for result in results:
    box = result['box']
    label = result['label']
    score = result['score']
    box['label'] = label
    box['score'] = score
    boxes.append(box)
    print(f"Detected {label} with confidence {score:.2f}")

# Draw bounding boxes and labels on the image 
for box in boxes:
    cv2.putText(image_cv, f"{box['label']}: {box['score']:.2f}", (box["xmin"], box["ymin"] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.rectangle(image_cv, (box["xmin"], box["ymin"]), (box["xmax"], box["ymax"]), (255, 0, 0), 2)

# Resize the image for better visualization
resized_image = cv2.resize(image_cv, (800, 600), interpolation=cv2.INTER_AREA)

# Show the image with detected objects
cv2.imshow("Detected Objects", resized_image)


# Wait for a key press and close the image window
cv2.waitKey(0)
cv2.destroyAllWindows()