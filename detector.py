from PIL import Image
import numpy as np
import cv2
import hashlib
from skimage.feature import local_binary_pattern


def generate_signature(hist):

    text=",".join(
        [str(round(x,5))
        for x in hist]
    )

    signature=hashlib.sha256(
        text.encode()
    ).hexdigest()

    return signature[:32]


def detect_image(uploaded_image):

    image=Image.open(
        uploaded_image
    ).convert("RGB")

    img=np.array(image)

    gray=cv2.cvtColor(
        img,
        cv2.COLOR_RGB2GRAY
    )

    gray=cv2.resize(
        gray,
        (256,256)
    )

    # texture fingerprint
    lbp=local_binary_pattern(
        gray,
        P=16,
        R=2,
        method="uniform"
    )

    hist,_=np.histogram(
        lbp.ravel(),
        bins=25,
        range=(0,25)
    )

    hist=hist.astype("float")

    hist=hist/(
        hist.sum()+1e-7
    )

    variance=np.var(hist)

    score=round(
        (1-min(
        variance*500,
        .90
        ))*100,
        2
    )

    signature=generate_signature(
        hist
    )

    if score>=75:

        status="Real Human"

    else:

        status="AI / Edited"

    return (
        status,
        score,
        signature,
        image
    )
