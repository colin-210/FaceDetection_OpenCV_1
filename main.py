import streamlit as st
import cv2
import numpy as np
from io import BytesIO
import base64
from PIL import Image

st.title("OpenCV Deep Learning based Face Detection")

image_file_buffer = st.file_uploader("Choose a file", type=['jpg', 'jpeg', 'png'])


# function for detecting faces in an image
def detectFaceOpenCVDnn(net, frame):
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
    net.setInput(blob)
    detections = net.forward()
    return detections


# function for adding bounding boxes around faces
def process_detections(frame, detections, conf_threshold):
    bboxes = []
    frame_h = frame.shape[0]
    frame_w = frame.shape[1]

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frame_w)
            y1 = int(detections[0, 0, i, 4] * frame_h)
            x2 = int(detections[0, 0, i, 5] * frame_w)
            y2 = int(detections[0, 0, i, 6] * frame_h)
            bboxes.append([x1, y1, x2, y2])
            bb_line_thickness = max(1, int(round(frame_h / 200)))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), bb_line_thickness, cv2.LINE_8)
    return frame, bboxes


# function to load dnn model
#@st.cache_resource
def load_model():
    modelFile = "res10_300X300_ssd_iter_140000_fp16.caffemodel"
    configFile = "deploy.prototxt"
    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
    return net


# function to generate download link
def get_image_download_link(img, filename, text):
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/txt;base64,{img_str}" download="{filename}">{text}</a>'
    return href


net = load_model()


if image_file_buffer is not None:
    # read file and convert to opencv image
    raw_bytes = np.asarray(bytearray(image_file_buffer.read()), dtype=np.uint8)
    # load image in BRG
    image = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)

    # create placeholders to display input and output images
    placeholders = st.columns(2)
    placeholders[0].text("Input Image")
    placeholders[0].image(image, channels='BRG')

    # create slider and get threshold
    conf_threshold = st.slider("Set Confidence Threshold", min_value=0.0, max_value=1.0, step=.01, value=0.5)

    # call the face detection model
    detections = detectFaceOpenCVDnn(net, image)

    # process the image
    out_img, _ = process_detections(image, detections, conf_threshold=conf_threshold)

    # display output image
    placeholders[1].text("Output Image")
    placeholders[1].image(out_img, channels='BRG')

    # convert opencv image to PIL
    out_image = Image.fromarray(out_img[:, :, ::-1])
    # download link for output image
    st.markdown(get_image_download_link(out_image, "face_output.jpg", "Download Output Image"), unsafe_allow_html=True)
