import os

from pycocotools.coco import COCO
import skimage.io as io
import matplotlib

from data.data_loader import DataLoader

PATH = os.path.dirname(os.path.abspath(__file__))
matplotlib.use('TkAgg')


class CocoLoader(DataLoader):

    def _load(self):
        data_type = 'val2017'
        annotation_file = os.path.join(PATH, 'annotations/instances_{}.json'.format(data_type))

        # initialize COCO api for instance annotations
        coco = COCO(annotation_file)

        # get all images containing given categories, select one at random
        cat_ids = coco.getCatIds(catNms=['dog'])
        img_ids = coco.getImgIds(catIds=cat_ids)

        print('Loading images... This may take a while.')
        images = []
        for img_id in img_ids:
            img_data = coco.loadImgs(img_id)[0]
            image = io.imread(img_data['coco_url'])
            images.append(image)

        return images
