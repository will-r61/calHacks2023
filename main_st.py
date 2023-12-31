from user_input import main
from intake_camera_data import main as search
import streamlit as st
import os
import cv2
import requests
from PIL import Image
import groundingdino.datasets.transforms as T
from groundingdino.util.inference import load_model, load_image, predict, annotate


st.title("Welcome to the Vision Voyager Demo!")


input_type = st.radio('How would you like to describe your object?', ['Text', 'Image'])
st.session_state['input_type'] = input_type

def get_image_path(img):
    # Create a directory and save the uploaded image.
    file_path = f"data/uploadedImages/{img.name}"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as img_file:
        img_file.write(img.getbuffer())
    return file_path

if st.session_state['input_type'] == "Image":
    uploaded_file = st.file_uploader("Upload your image here!")
    if 'uploaded_file' not in st.session_state:
        st.session_state['uploaded_file'] = uploaded_file

        bytes_data = get_image_path(uploaded_file)

        image = Image.frombytes('RGBA', (128,128), bytes_data)

        st.session_state['object_description'] = image


if st.session_state['input_type'] == "Text":
    object_description = st.text_input("Describe your object here!")
    if 'object_description' not in st.session_state:
        st.session_state['object_description'] = object_description

#item_description = main()
start_initialized = st.button('Start demo!')

#if start_initialized:
#    search()

if start_initialized:

    text_prompt = st.session_state['object_description']

    HOME = os.getcwd()
    # set model configuration file path
    # CONFIG_PATH = os.path.join(HOME, "groundingdino/config/GroundingDINO_SwinT_OGC.py")

    # set model weight file ath
    WEIGHTS_PATH = 'weights/groundingdino_swint_ogc.pth'

    # set text prompt
    TEXT_PROMPT = text_prompt

    # set box and text threshold values
    BOX_TRESHOLD = 0.3
    TEXT_TRESHOLD = 0.2
    PREDICTION_THRESHOLD = 0.3

    model = load_model("GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", "GroundingDINO/weights/groundingdino_swint_ogc.pth")

    item_not_found = True

    while item_not_found:
        # ESP32 URL
        URL = 'http://172.20.10.6'
        AWB = True

        # Face recognition and opencv setup
        cap = cv2.VideoCapture(URL + ":81/stream")

        requests.get(URL + "/control?var=framesize&val={}".format(10))
        requests.get(URL + "/control?var=quality&val={}".format(10))
        
        ret, frame = cap.read()
        print('reading from stream')
        # create a transform function by applying 3 image transaformations
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
        )
        # convert frame to a PIL object in RGB space
        image_source = Image.fromarray(frame).convert("RGB")
        # convert the PIL image object to a transform object
        image_transformed, _ = transform(image_source, None)
        
        image_source.save('test.jpg')
        st.image('test.jpg', 'Captured Image')

        # predict boxes, logits, phrases
        print(TEXT_PROMPT)
        boxes, logits, phrases = predict(
        model=model, 
        image=image_transformed, 
        caption=TEXT_PROMPT, 
        box_threshold=BOX_TRESHOLD, 
        text_threshold=TEXT_TRESHOLD,
        device='cpu')

        # annotate the image
        annotated_frame = annotate(image_source=frame, boxes=boxes, logits=logits, phrases=phrases)
        # display the output
        out_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        cv2.imshow('frame', out_frame)
        
        #save the output to JPG
        cv2.imwrite('model_output.jpg', out_frame)
        st.image('model_output.jpg', f'Identified Objects, Confidence: {logits}')

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if logits.numel() == 0:
            pass
        else:
            for value in logits.view(-1):
                ## TODO find highest value if not too lazy
                if value > PREDICTION_THRESHOLD:
                    item_not_found = False
                    print("Item found!")
                    print("Item is: " + TEXT_PROMPT)
                    print("Confidence: " + str(value))
                    #print("Box: " + str(boxes))
                    print("Text: " + str(phrases))
                    cv2.imwrite('item_found.jpg', out_frame)
                    st.success('Item Found!')
                    break
            if item_not_found == False:
                break
    cap.release()
    cv2.destroyAllWindows()
