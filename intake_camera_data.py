import os
import cv2
import numpy as np
import torch
from PIL import Image
import groundingdino.datasets.transforms as T
from groundingdino.util.inference import load_model, load_image, predict, annotate


def main(text_prompt):

    HOME = os.getcwd()
    # set model configuration file path
    # CONFIG_PATH = os.path.join(HOME, "groundingdino/config/GroundingDINO_SwinT_OGC.py")

    # set model weight file ath
    WEIGHTS_PATH = 'GroundingDINO/weights/groundingdino_swint_ogc.pth'

    # set text prompt
    TEXT_PROMPT = text_prompt

    # set box and text threshold values
    BOX_TRESHOLD = 0.35
    TEXT_TRESHOLD = 0.25
    PREDICTION_THRESHOLD = 0.6

    model = load_model("GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", "GroundingDINO/weights/groundingdino_swint_ogc.pth")

    item_not_found = True

    while item_not_found:
        # ESP32 URL
        URL = 'http://172.20.10.6'
        AWB = True

        # Face recognition and opencv setup
        cap = cv2.VideoCapture(URL + ":81/stream")
        
        ret, frame = cap.read()
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

        # predict boxes, logits, phrases
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
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        print(logits)

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
                    break
            if item_not_found == False:
                break
    cap.release()
    cv2.destroyAllWindows()


