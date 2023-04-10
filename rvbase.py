from rastervision.core.data import ClassConfig
from rastervision.pytorch_learner import (
    ObjectDetectionSlidingWindowGeoDataset,
)
from rastervision.pytorch_learner.object_detection_utils import collate_fn
from torch.utils.data import DataLoader
import pathlib
import boto3
    
class RVBase:
    
    def __init__(self, tr_uris, val_uris, pred_uris, output, class_config, kw=None):
        self.train_uris = tr_uris
        self.val_uris = val_uris
        self.pred_uris = pred_uris
        self.cc = ClassConfig(
            names=class_config["names"], 
            colors=class_config["colors"],
        )
        self.output_uri = output.get("uri")
        self.bucket = output.get("bucket")
        self.kw = kw
        
    def build_train_ds(self, **dskw):
        kw = self.kw.get("train_data_kw", {})
        train_ds = ObjectDetectionSlidingWindowGeoDataset.from_uris(
            class_config=self.cc,
            image_uri=self.train_uris[0],
            aoi_uri=self.train_uris[2],
            label_vector_uri=self.train_uris[1],
            label_vector_default_class_id=self.cc.get_class_id('DF'),
            size=kw.get("size", 325),
            stride=kw.get("stride", 325),
            **dskw)
        return train_ds
    
    def build_val_ds(self, **dskw):
        kw = self.kw.get("val_data_kw", {})
        val_ds = ObjectDetectionSlidingWindowGeoDataset.from_uris(
            class_config=self.cc,
            image_uri=self.val_uris[0],
            aoi_uri=self.val_uris[2],
            label_vector_uri=self.val_uris[1],
            label_vector_default_class_id=self.cc.get_class_id('DF'),
            size=kw.get("size", 325),
            stride=kw.get("stride", 325),
            **dskw)
        return val_ds
    
    def resolve_pred_uri(self):
        kw = self.kw.get("pred_data_kw", {})
        uri = self.pred_uris[0]
        if pathlib.Path(uri).is_file():
            return uri
        if isinstance(uri, list):
            return uri[:kw.get("num_images")]
        def get_tiles_from_dir(d):
            v = []
            for x in pathlib.Path(d).glob("*.tif"):
                v.append(x.as_posix())
            return v
        def get_tiles_from_s3(p):
            s3 = boto3.resource("s3")
            bucket = s3.Bucket(self.bucket)
            uris = []
            for obj in bucket.objects.filter(Prefix=p):
                uris.append("s3://" + obj.bucket_name + "/" + obj.key)
            return uris
        if uri[:3] == "s3:":
            pref = pathlib.Path(uri)
            pref = pathlib.Path(*pref.parts[2:]).as_posix()
            img_uri = get_tiles_from_s3(pref)
        else:
            img_uri = get_tiles_from_dir(uri)
        if kw.get("num_images", None) is not None:
            img_uri = img_uri[:kw.get("num_images")]
        return img_uri
    
    def build_pred_ds(self, **dskw):
        kw = self.kw.get("pred_data_kw", {})
        img_uri = self.resolve_pred_uri()
        pred_ds = ObjectDetectionSlidingWindowGeoDataset.from_uris(
            class_config=self.cc,
            image_uri=img_uri,
            size=kw.get("size", 325),
            stride=kw.get("stride", 325),
            **dskw)
        return pred_ds
    
    def build_train_loader(self, **dskw):
        tds = self.build_train_ds(**dskw)
        tkw = self.kw.get("train_data_kw", {})
        train = DataLoader(tds, batch_size=tkw.get("batch_size", 2), shuffle=True, 
                           num_workers=tkw.get("num_workers", 4), collate_fn=collate_fn)
        return train
    
    def build_val_loader(self, **dskw):
        vds = self.build_val_ds(**dskw)
        vkw = self.kw.get("val_data_kw", {})
        val = DataLoader(vds, batch_size=vkw.get("batch_size", 2), 
                         num_workers=vkw.get("num_workers", 4), collate_fn=collate_fn)
        return val
    
    def build_pred_loader(self, **dskw):
        kw = self.kw.get("pred_data_kw", {})
        pds = self.build_pred_ds(**dskw)
        pred_dl = DataLoader(pds, 
                      batch_size=kw.get("batch_size", 2), num_workers=kw.get("num_workers", 4))
        return pred_dl
    
    def run(self):
        if self.kw.get("train", False):
            self.train()
        if self.kw.get("predict", False):
            self.predict()