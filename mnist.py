from fastai.vision import *
from fastai.metrics import error_rate

INPUT_DATASET_URL = URLs.MNIST_SAMPLE

path = untar_data(INPUT_DATASET_URL); 
print(path)

tfms = get_transforms(do_flip=False)
data = ImageDataBunch.from_folder(path, ds_tfms=tfms, size=26)

data.show_batch(rows=3, figsize=(5,5))

learn = cnn_learner(data, models.resnet18, metrics=accuracy)
learn.fit(2)
