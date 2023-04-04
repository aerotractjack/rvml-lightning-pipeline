import albumentations as A
from rastervision.pytorch_learner import (
    SemanticSegmentationRandomWindowGeoDataset,
    SemanticSegmentationSlidingWindowGeoDataset,
    SemanticSegmentationVisualizer)
from rastervision.core.data import ClassConfig
from tqdm.autonotebook import tqdm
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torchvision.models.segmentation import deeplabv3_resnet50
import pytorch_lightning as pl
from rastervision.pipeline.file_system import make_dir
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from rastervision.core.data import SemanticSegmentationLabels
from rastervision.core.data.label_store.semantic_segmentation_label_store_config import PolygonVectorOutputConfig
import pathlib
import geopandas
import rasterio
from rasterio.features import shapes
from smart_open import open as smart_open
import io
import boto3

class SemanticSegmentation(pl.LightningModule):
    
    def __init__(self, deeplab, lr=1e-4):
        super().__init__()
        self.deeplab = deeplab
        self.lr = lr

    def forward(self, img):
        return self.deeplab(img)['out']

    def training_step(self, batch, batch_idx):
        img, mask = batch
        img = img.float()
        mask = mask.long()
        out = self.forward(img)
        loss = F.cross_entropy(out, mask)
        log_dict = {'train_loss': loss}
        self.log_dict(log_dict, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        img, mask = batch
        img = img.float()
        mask = mask.long()
        out = self.forward(img)
        loss = F.cross_entropy(out, mask)
        log_dict = {'validation_loss': loss}
        self.log_dict(log_dict, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr)
        return optimizer
    
class RVLightning:
    
    def __init__(self, tr_uris, val_uris, pred_uris, output, class_config, kw=None):
        self.train_uris = tr_uris
        self.val_uris = val_uris
        self.pred_uris = pred_uris
        self.cc = ClassConfig(
            names=class_config["names"], 
            colors=class_config["colors"],
            null_class="null")
        self.output_uri = output.get("uri")
        self.bucket = output.get("bucket")
        self.kw = kw
        
    def build_train_ds(self):
        kw = self.kw.get("train_ds_kw", {})
        data_augmentation_transform = A.Compose([
            A.Flip(),
            A.ShiftScaleRotate(),
            A.RGBShift()
        ])
        train_ds = SemanticSegmentationRandomWindowGeoDataset.from_uris(
            class_config=self.cc,
            image_uri=self.train_uris[0],
            aoi_uri=self.train_uris[2],
            label_vector_uri=self.train_uris[1],
            label_vector_default_class_id=self.cc.get_class_id('DF'),
            size_lims=kw.get("size_lims", [300,350]),
            out_size=kw.get("out_size", 325),
            max_windows=kw.get("max_windows", 10),
            transform=data_augmentation_transform)
        return train_ds
    
    def build_val_ds(self):
        kw = self.kw.get("val_ds_kw", {})
        val_ds = SemanticSegmentationSlidingWindowGeoDataset.from_uris(
            class_config=self.cc,
            image_uri=self.val_uris[0],
            aoi_uri=self.val_uris[2],
            label_vector_uri=self.val_uris[1],
            label_vector_default_class_id=self.cc.get_class_id('DF'),
            size=kw.get("size", 325),
            stride=kw.get("stride", 325))
        return val_ds
    
    def build_pred_ds(self):
        kw = self.kw.get("pred_ds_kw", {})
        def get_tiles_from_dir(d):
            v = [x.as_posix() for x in pathlib.Path(d).glob("*.tif")]
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
        pred_ds = SemanticSegmentationSlidingWindowGeoDataset.from_uris(
            class_config=self.cc,
            image_uri=img_uri,
            aoi_uri=self.pred_uris[1],
            size=kw.get("size", 325),
            stride=kw.get("stride", 325))
        return pred_ds
    
    def train(self):
        kw = self.kw.get("train_kw", {})
        batch_size = kw.get("batch_size", 8)
        lr = float(kw.get("lr", 1e-4))
        epochs = kw.get("epochs", 1)
        output_dir = self.output_uri
        make_dir(output_dir)
        fast_dev_run = False
        deeplab = deeplabv3_resnet50(num_classes=len(self.cc) + 1)
        model = SemanticSegmentation(deeplab, lr=lr)
        tb_logger = TensorBoardLogger(save_dir=output_dir + "/tensorboard", flush_secs=10)
        trainer = pl.Trainer(
            accelerator='auto',
            min_epochs=1,
            max_epochs=epochs+1,
            default_root_dir=output_dir + "/trainer",
            logger=[tb_logger],
            fast_dev_run=fast_dev_run,
            log_every_n_steps=1,
        )
        tds = self.build_train_ds()
        vds = self.build_val_ds()
        train_dl = DataLoader(tds, batch_size=batch_size, shuffle=True, num_workers=4)
        val_dl = DataLoader(vds, batch_size=batch_size, num_workers=4)
        trainer.fit(model, train_dl, val_dl)
        trainer.save_checkpoint(output_dir + "/trainer/final-model.ckpt")
        
    def prediction_iterator(self, pred_dl, model, n=None):
        for i, (x, _) in tqdm(enumerate(pred_dl)):
            if n is not None and i >= n:
                break
            with torch.inference_mode():
                out_batch = model(x)
            for out in out_batch:
                yield out.numpy()
        
    def predict(self):
        kw = self.kw.get("predict_kw", {})
        def fix_keys(data):
            for _ in range(len(data)):
                k, v = data.popitem(False)
                newk = k[k.index(".")+1:]
                data[newk] = v
            return data
        ckpt_path = f'{self.output_uri}/trainer/final-model.ckpt'
        with smart_open(ckpt_path, 'rb') as f:
            buffer = io.BytesIO(f.read())
            ckpt = torch.load(buffer)
        state_dict = ckpt["state_dict"]
        state_dict = fix_keys(state_dict)
        deeplab = deeplabv3_resnet50(num_classes=len(self.cc) + 1)
        deeplab.load_state_dict(state_dict)
        model = SemanticSegmentation(deeplab)
        pds = self.build_pred_ds()
        pred_dl = DataLoader(pds, batch_size=8, num_workers=4)
        predictions = self.prediction_iterator(pred_dl, model, kw.get("num_preds"))
        pred_labels = SemanticSegmentationLabels.from_predictions(
            pds.windows,
            predictions,
            smooth=True,
            extent=pds.scene.extent,
            num_classes=len(self.cc)+1
        )
        if kw.get("polygon_output", True):
            voutputs = [
                PolygonVectorOutputConfig(
                    class_id=0,
                    uri=f"{self.output_uri}/pred-labels-vectors/{0}.geojson",
                    denoise=kw.get("denoise", 10),
                ) 
            ]
        else:
            voutputs = []
        pred_labels.save(
            uri=f"{self.output_uri}/pred-labels-scores-vectors",
            crs_transformer=pds.scene.raster_source.crs_transformer,
            class_config=self.cc,
            discrete_output=False,
            smooth_output=False,
            smooth_as_uint8=False,
            vector_outputs=voutputs,
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

    # obj.train()

    return obj.predict()

if __name__ == "__main__":
    import sys
    run(sys.argv[1])