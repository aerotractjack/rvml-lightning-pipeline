import albumentations as A
from rastervision.core.data import ClassConfig
from rastervision.pytorch_learner import (
    ObjectDetectionRandomWindowGeoDataset,
    ObjectDetectionSlidingWindowGeoDataset,
)
from rastervision.pytorch_learner.object_detection_utils import (
   TorchVisionODAdapter, compute_coco_eval, collate_fn, BoxList)
from rastervision.core.data.label.object_detection_labels import ObjectDetectionLabels
from rastervision.pipeline.file_system import make_dir
from torch.utils.data import DataLoader
from torchvision.models import detection
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm
from smart_open import smart_open
import io
import pathlib
import boto3

def exit():
    import sys
    sys.exit(0)

class ObjectDetection(pl.LightningModule):

    def __init__(self, backbone, lr=1e-4):
        super().__init__()
        self.backbone = TorchVisionODAdapter(backbone)
        self.lr = lr
        self.val_map_metric = MeanAveragePrecision(box_format="xyxy", iou_thresholds=[0.5])

    def to_device(self, x, device):
        if isinstance(x, list):
            return [_x.to(device) if _x is not None else _x for _x in x]
        else:
            return x.to(device)

    def forward(self, img):
        return self.backbone(img)
    
    def output_to_numpy(self, out, class_id_key="class_ids"):
        def boxlist_to_numpy(boxlist):
            npy = {}
            npy["boxes"] = boxlist.convert_boxes('xyxy').cpu().float()
            # npy["class_ids"] = boxlist.get_field('class_ids').cpu()
            npy[class_id_key] = boxlist.get_field('class_ids').cpu().int()
            scores = boxlist.get_field('scores')
            if scores is not None:
                npy["scores"] = scores.float()
            return npy
        if isinstance(out, BoxList):
            return boxlist_to_numpy(out)
        else:
            return [boxlist_to_numpy(boxlist) for boxlist in out]
    
    def training_step(self, batch, batch_ind):
        x, y = batch
        loss_dict = self.backbone(x, y)
        loss_dict['loss'] = sum(loss_dict.values())
        self.log("train_loss", loss_dict["loss"])
        return loss_dict

    def validation_step(self, batch, batch_ind):
        x, ys = batch
        outs = self.backbone(x)
        ys = self.output_to_numpy(ys, class_id_key="labels")
        outs = self.output_to_numpy(outs, class_id_key="labels")
        self.val_map_metric(outs, ys)
        self.log_dict(self.val_map_metric.compute())
        return {'ys': ys, 'outs': outs}

    def on_validation_batch_end(self, out, batch, batch_idx):
        self.log_dict(self.val_map_metric.compute())
    
    def predict(self, x, raw_out=False, out_shape=None):
        self.backbone.eval()
        with torch.no_grad():
            device = torch.device('cuda')
            out_batch = self.backbone.to(device)(x.to(device))
        if out_shape is None:
            return out_batch
        h_in, w_in = x.shape[-2:]
        h_out, w_out = out_shape
        yscale, xscale = (h_out / h_in), (w_out / w_in)
        with torch.inference_mode():
            for out in out_batch:
                out.scale(yscale, xscale)
        return out_batch

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.backbone.parameters(), lr=self.lr)
        return optimizer
    
class RVLightning:
    
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
        
    def build_train_ds(self):
        kw = self.kw.get("train_data_kw", {})
        train_ds = ObjectDetectionSlidingWindowGeoDataset.from_uris(
            class_config=self.cc,
            image_uri=self.train_uris[0],
            aoi_uri=self.train_uris[2],
            label_vector_uri=self.train_uris[1],
            label_vector_default_class_id=self.cc.get_class_id('DF'),
            size=kw.get("size", 325),
            stride=kw.get("stride", 325))
        return train_ds
    
    def build_val_ds(self):
        kw = self.kw.get("val_data_kw", {})
        val_ds = ObjectDetectionSlidingWindowGeoDataset.from_uris(
            class_config=self.cc,
            image_uri=self.val_uris[0],
            aoi_uri=self.val_uris[2],
            label_vector_uri=self.val_uris[1],
            label_vector_default_class_id=self.cc.get_class_id('DF'),
            size=kw.get("size", 325),
            stride=kw.get("stride", 325))
        return val_ds
    
    def build_pred_ds(self):
        kw = self.kw.get("pred_data_kw", {})
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
        if self.pred_uris[0][:3] == "s3:":
            pref = pathlib.Path(self.pred_uris[0])
            pref = pathlib.Path(*pref.parts[2:]).as_posix()
            img_uri = get_tiles_from_s3(pref)
        else:
            img_uri = get_tiles_from_dir(self.pred_uris[0])
        if kw.get("num_images", None) is not None:
            img_uri = img_uri[:kw.get("num_images")]
        pred_ds = ObjectDetectionSlidingWindowGeoDataset.from_uris(
            class_config=self.cc,
            image_uri=img_uri,
            size=kw.get("size", 325),
            stride=kw.get("stride", 325))
        return pred_ds
    
    def build_train_val_loader(self):
        tds, vds = self.build_train_ds(), self.build_val_ds()
        tkw = self.kw.get("train_data_kw", {})
        vkw = self.kw.get("val_data_kw", {})
        train = DataLoader(tds, batch_size=tkw.get("batch_size", 2), shuffle=True, 
                           num_workers=tkw.get("num_workers", 4), collate_fn=collate_fn)
        val = DataLoader(vds, batch_size=vkw.get("batch_size", 2), 
                         num_workers=vkw.get("num_workers", 4), collate_fn=collate_fn)
        return train, val
    
    def build_pred_loader(self):
        kw = self.kw.get("pred_data_kw", {})
        pds = self.build_pred_ds()
        pred_dl = DataLoader(pds, 
                      batch_size=kw.get("batch_size", 2), num_workers=kw.get("num_workers", 4))
        return pred_dl
    
    def model_type(self):
        return getattr(detection, self.kw["model_kw"]["name"])
    
    def build_model(self):
        kw = self.kw.get("model_kw", {})
        lr = float(kw.get("lr", 1e-4))
        output_dir = self.output_uri
        make_dir(output_dir)
        backbone = self.model_type()(
            num_classes=len(self.cc)+1, pretrained=False)
        model = ObjectDetection(backbone, lr=lr)
        return model
    
    def train(self):
        kw = self.kw.get("train_kw", {})
        if not kw.get("run_training", True):
            return
        epochs = kw.get("epochs", 1)
        model = self.build_model()
        output_dir = self.output_uri
        tb_logger = TensorBoardLogger(save_dir=output_dir + "/tensorboard", flush_secs=10)
        trainer = pl.Trainer(
            accelerator='auto',
            min_epochs=1,
            max_epochs=epochs+1,
            default_root_dir=output_dir + "/trainer",
            logger=[tb_logger],
            fast_dev_run=False,
            log_every_n_steps=1,
            callbacks=[EarlyStopping(monitor="train_loss", mode="min")],
        )
        train_dl, val_dl = self.build_train_val_loader()
        trainer.fit(model, train_dl, val_dl)
        trainer.save_checkpoint(output_dir + "/trainer/final-model.ckpt")

    def prediction_iterator(self, pred_dl, model, n=None):
        for i, (x, _) in tqdm(enumerate(pred_dl)):
            if n is not None and n != "None" and n > 0 and i >= n:
                break
            with torch.inference_mode():
                out_batch = model.predict(x)
                out_batch = model.output_to_numpy(out_batch, class_id_key="class_ids")
            for out in out_batch:
                yield out
                
    def reconstruct_model(self):
        def fix_keys(data):
            for _ in range(len(data)):
                k, v = data.popitem(False)
                newk = k.split(".")
                newk = ".".join(newk[2:])
                data[newk] = v
            return data
        ckpt_path = f'{self.output_uri}/trainer/final-model.ckpt'
        with smart_open(ckpt_path, 'rb') as f:
            buffer = io.BytesIO(f.read())
            ckpt = torch.load(buffer)
        state_dict = ckpt["state_dict"]
        state_dict = fix_keys(state_dict)
        deeplab = self.model_type()(num_classes=len(self.cc) + 1)
        deeplab.load_state_dict(state_dict)
        model = ObjectDetection(deeplab)
        return model
    
    def predict(self):
        kw = self.kw.get("predict_kw", {})
        model = self.reconstruct_model()
        pred_dl = self.build_pred_loader()
        predictions = self.prediction_iterator(pred_dl, model, kw.get("max_images", None))
        pred_labels = ObjectDetectionLabels.from_predictions(
            pred_dl.dataset.windows,
            predictions,
        )
        pred_labels.save(
            uri=f"{self.output_uri}/pred-labels.geojson",
            crs_transformer=pred_dl.dataset.scene.raster_source.crs_transformer,
            class_config=self.cc,
        )
        return pred_labels
    
def run(config_path):
    from configreader import yaml2dict

    conf = yaml2dict(config_path)

    obj = RVLightning(
        conf["train_uri"],
        conf["val_uri"],
        conf["pred_uri"],
        conf["output"],
        conf["class_config"],
        conf
    )

    obj.train()
    obj.predict()

if __name__ == "__main__":
    import sys
    file = "input/giustina230OG.yaml"
    if len(sys.argv) > 1:
        file = sys.argv[1]
    run(file)