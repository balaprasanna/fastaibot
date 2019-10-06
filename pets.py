from fastai.vision import *
from fastai.metrics import error_rate

INPUT_DATASET_URL = URLs.PETS

bs = 64
# bs = 16   # uncomment this line if you run out of memory even after clicking Kernel->Restart

path = untar_data(INPUT_DATASET_URL ); path

path.ls()

path_anno = path/'annotations'
path_img = path/'images'


fnames = get_image_files(path_img)
fnames[:5]

np.random.seed(2)
pat = r'/([^/]+)_\d+.jpg$'

data = ImageDataBunch.from_name_re(path_img, fnames, pat, ds_tfms=get_transforms(), size=224, bs=bs
                                  ).normalize(imagenet_stats)
data.show_batch(rows=3, figsize=(7,6))

print(data.classes)
len(data.classes),data.c

learn = cnn_learner(data, models.resnet34, metrics=error_rate)

learn.model

learn.fit_one_cycle(4)

learn.save('stage-1')

interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()

len(data.valid_ds)==len(losses)==len(idxs)


interp.plot_top_losses(9, figsize=(15,11))

doc(interp.plot_top_losses)


interp.plot_confusion_matrix(figsize=(12,12), dpi=60)

interp.most_confused(min_val=2)

learn.load('stage-1');
learn.lr_find()
learn.recorder.plot()

learn.unfreeze()
learn.fit_one_cycle(2, max_lr=slice(1e-6,1e-4))