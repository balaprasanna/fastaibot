from fastai.vision import *
from fastai.metrics import error_rate

INPUT_DATASET_URL = URLs.PETS

bs = 64
# bs = 16   # uncomment this line if you run out of memory even after clicking Kernel->Restart

path = untar_data(INPUT_DATASET_URL); 
print(path)
#path.ls()

path_anno = path/'annotations'
path_img = path/'images'

fnames = get_image_files(path_img)

np.random.seed(2)
pat = r'/([^/]+)_\d+.jpg$'


data = ImageDataBunch.from_name_re(path_img, fnames, pat, ds_tfms=get_transforms(),
                                   size=299, bs=bs//2).normalize(imagenet_stats)

learn = cnn_learner(data, models.resnet50, metrics=error_rate)

# learn.fit_one_cycle(1)
# learn.save('stage-1-50')
learn.load('stage-1-50');

learn.lr_find()
learn.recorder.plot()
sys.exit(0)

learn.unfreeze()
learn.fit_one_cycle(3, max_lr=slice(1e-6,1e-4))

learn.load('stage-1-50');

interp = ClassificationInterpretation.from_learner(learn)

interp.most_confused(min_val=2)