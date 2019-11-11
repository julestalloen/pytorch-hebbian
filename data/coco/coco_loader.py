import os
import logging

from pycocotools.coco import COCO
import skimage.io as io
import matplotlib

from data.data_loader import ImageLoader

PATH = os.path.dirname(os.path.abspath(__file__))
matplotlib.use('TkAgg')


class CocoLoader(ImageLoader):

    def __init__(self, annotation_type: str = 'val2017', categories: list = None):
        self.annotation_path = os.path.join(PATH, 'annotations/instances_{}.json'.format(annotation_type))
        self.coco = COCO(self.annotation_path)

        if categories is None:
            self.categories = ['dog', 'cat', 'person']
        else:
            self.categories = categories

    def _load(self):
        cat_ids = self.coco.getCatIds(catNms=self.categories)

        img_ids = []
        for cat_id in cat_ids:
            img_ids.extend(self.coco.getImgIds(catIds=[cat_id]))

        logging.info('Retrieved {} image IDs'.format(len(img_ids)))

        print('Loading images... This may take a while.')
        images = []
        for img_id in img_ids:
            img_data = self.coco.loadImgs(img_id)[0]
            image = io.imread(img_data['coco_url'])
            images.append(image)

        return images
