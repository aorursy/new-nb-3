image_fn = "/kaggle/input/siim-isic-melanoma-classification/jpeg/train/ISIC_0077735.jpg"

small_image_fn = "small.jpg"

import io

f = open(image_fn, "rb")

data = f.read()

print(len(data))


import PIL.Image as Image

image = Image.open(io.BytesIO(data))

down = image.resize((224, 224))

down.save(small_image_fn)
from IPython.display import Image

Image(image_fn)
Image(small_image_fn)