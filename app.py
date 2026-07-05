import streamlit as st
import numpy as np
import cv2
import keras
from PIL import Image

st.set_page_config(page_title="Facial Emotion Recognition", page_icon="🎭", layout="centered")

MODEL_DIR = "model"
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]


@st.cache_resource
def load_model():
    with open(f"{MODEL_DIR}/config.json") as f:
        config_json = f.read()
    model = keras.models.model_from_json(config_json)
    model.load_weights(f"{MODEL_DIR}/model.weights.h5")
    return model


@st.cache_resource
def load_face_detector():
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    return cv2.CascadeClassifier(cascade_path)


def predict_face(model, gray_face_48x48):
    """gray_face_48x48: uint8/float array of shape (48,48), raw pixel values 0-255.
    The model has a built-in Rescaling layer, so do NOT divide by 255 here."""
    face_input = gray_face_48x48.astype("float32")
    face_input = np.expand_dims(face_input, axis=-1)  # (48,48,1)
    face_input = np.expand_dims(face_input, axis=0)    # (1,48,48,1)
    preds = model.predict(face_input, verbose=0)[0]
    return preds


model = load_model()
face_cascade = load_face_detector()

st.title("🎭 Facial Emotion Recognition")
st.write(
    "Take a photo or upload an image. The app detects faces and predicts "
    "the emotion for each one using your trained CNN."
)

tab1, tab2 = st.tabs(["📷 Camera", "📁 Upload"])

image_source = None
with tab1:
    cam_image = st.camera_input("Take a picture")
    if cam_image is not None:
        image_source = Image.open(cam_image)

with tab2:
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded is not None:
        image_source = Image.open(uploaded)

if image_source is not None:
    img_array = np.array(image_source.convert("RGB"))
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48)
    )

    if len(faces) == 0:
        st.warning("No face detected. Try a clearer, front-facing, well-lit photo.")
    else:
        display_img = img_array.copy()
        results = []

        for (x, y, w, h) in faces:
            face_roi = gray[y:y + h, x:x + w]
            face_resized = cv2.resize(face_roi, (48, 48))
            preds = predict_face(model, face_resized)
            top_idx = int(np.argmax(preds))
            label = f"{EMOTIONS[top_idx]} ({preds[top_idx] * 100:.1f}%)"
            results.append(preds)

            cv2.rectangle(display_img, (x, y), (x + w, y + h), (0, 200, 0), 2)
            cv2.putText(
                display_img, label, (x, max(y - 10, 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2
            )

        st.image(display_img, caption=f"{len(faces)} face(s) detected", use_container_width=True)

        st.subheader("Emotion probabilities (first face)")
        st.bar_chart({EMOTIONS[i]: float(results[0][i]) for i in range(len(EMOTIONS))})
else:
    st.info("Use the Camera or Upload tab above to get started.")
