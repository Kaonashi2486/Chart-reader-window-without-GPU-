import os
import json
import numpy as np
import pickle
import copy
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

class Chart:
    def __init__(self, db_config, split):
        # Use cross-platform path handling
        data_dir = os.path.join("data")
        cache_dir = os.path.join("cache")
        
        self._split = split
        self._dataset = {
            "trainchart": "train",
            "valchart": "val",
            "testchart": "test"
        }[self._split]
        is_inference = False
        self._coco_dir = data_dir

        # Handle paths
        self._label_dir = os.path.join(self._coco_dir, "annotations")
        self._label_file = os.path.join(self._label_dir, f"{self._dataset}.json")

        if not os.path.exists(self._label_file):
            self._label_file = None
            is_inference = True
        print(f"Label file: {self._label_file}")

        self._image_dir = os.path.join(self._coco_dir, "images", self._dataset)
        self._image_file = os.path.join(self._image_dir, "{}")

        self._mean = np.array([0.40789654, 0.44719302, 0.47026115], dtype=np.float32)
        self._std = np.array([0.28863828, 0.27408164, 0.27809835], dtype=np.float32)

        # Initialize cat IDs and classes as before
        self._cat_ids = [0, 1, 2]
        self._classes = {ind: cat_id for ind, cat_id in enumerate(self._cat_ids)}
        self._coco_to_class_map = {value: key for key, value in self._classes.items()}

        self._cache_file = os.path.join(cache_dir, f"{self._dataset}_cache.pkl")
        
        if not is_inference:
            self._load_data()
            self._db_inds = np.arange(len(self._image_ids))
            self._load_coco_data()

    def _load_data(self):
        if not os.path.exists("./cache"):
            os.makedirs("./cache")
        print(f"Loading from cache file: {self._cache_file}")
        if not os.path.exists(self._cache_file):
            print("No cache file found...")
            self._extract_data()
            with open(self._cache_file, "wb") as f:
                pickle.dump([self._detections, self._image_ids], f)
        else:
            with open(self._cache_file, "rb") as f:
                self._detections, self._image_ids = pickle.load(f)

    def _load_coco_data(self):
        self._coco = COCO(self._label_file)
        with open(self._label_file, "r") as f:
            data = json.load(f)

        coco_ids = self._coco.getImgIds()
        eval_ids = {
            self._coco.loadImgs(coco_id)[0]["file_name"]: coco_id
            for coco_id in coco_ids
        }

        self._coco_categories = data["categories"]
        self._coco_eval_ids = eval_ids

    # Keep the rest of the class the same with minor adjustments for CPU processing if needed
