# Facial Emotion Recognition — Streamlit Demo

A ready-to-run web app for your CNN facial emotion recognition model
(48x48 grayscale input, 7-class softmax: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral).

link: https://facial-emotion-app-iskqktrh2rstzshrgctvgu.streamlit.app/

## Folder structure

```
emotion_app/
├── app.py              # Streamlit app
├── requirements.txt     # Python dependencies
├── packages.txt         # System packages (needed for OpenCV on Streamlit Cloud)
└── model/
    ├── config.json
    ├── model.weights.h5
    └── metadata.json
```


## How it works

1. Face detection uses OpenCV's Haar Cascade (`haarcascade_frontalface_default.xml`,
   bundled with `opencv-python-headless` — no extra download needed).
2. Each detected face is cropped, converted to grayscale, and resized to 48x48.
3. **Important:** your model already has a `Rescaling(1/255)` layer built in, so the
   app feeds raw pixel values (0–255) directly into the model — it does **not**
   divide by 255 again. If you ever swap in a different model that expects
   pre-normalized input, update `predict_face()` in `app.py` accordingly.
4. The model outputs a softmax over 7 emotions; the app shows the top prediction as
   a label on the image, plus a full probability bar chart for the first detected face.



- This uses `st.camera_input`, which captures a still photo per click (not continuous
  live video) — this is intentional, since Streamlit doesn't support real-time video
  streams natively without extra components. It's simpler and more reliable for a
  demo/viva than a live webcam feed.
- If you want a truly **live**, frame-by-frame webcam feed instead of snapshot-based,
  that needs `streamlit-webrtc` (a bit more setup) — let me know if you'd like that
  version instead.
- Multiple faces in one photo are all detected and boxed, but the probability chart
  shows only the first detected face for clarity.
