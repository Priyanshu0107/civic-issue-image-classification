import tensorflow as tf
import numpy as np
import cv2

# Load model
model = tf.keras.models.load_model("civic_issue_classifier.h5")

classes = ["garbage", "electricity", "road"]

def predict_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img)[0]
    confidence = np.max(preds)
    class_index = np.argmax(preds)

    if confidence < 0.60:
        return "Other", confidence
    else:
        return classes[class_index], confidence


# Test image
result, conf = predict_image("elec2.jpg")
print("Prediction:", result)
print("Confidence:", round(conf*100, 2), "%")
