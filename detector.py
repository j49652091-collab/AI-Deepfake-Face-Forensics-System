from PIL import Image
import numpy as np
import cv2


def detect_image(uploaded_image):

    image=Image.open(uploaded_image).convert("RGB")

    img=np.array(image)

    gray=cv2.cvtColor(
        img,
        cv2.COLOR_RGB2GRAY
    )

    brightness=np.mean(gray)

    if brightness>100:
        status="Likely Real"
        score=85
    else:
        status="Possibly AI / Edited"
        score=60

    return status,score,image
