from rastervision.core.data.label.object_detection_labels import ObjectDetectionLabels
from rastervision.pipeline.file_system import make_dir
import torch
from torchvision.models import detection
from torchvision import models
from torchvision.models import ResNet50_Weights
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from tqdm import tqdm
from smart_open import smart_open
import io
from lightningmodel import LightningModel
from rvbase import RVBase
from util import run_if_main, output_to_array
    
class RVLightning(RVBase):
    
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)

    def model_type(self):
        try:
            return getattr(detection, self.kw["model_kw"]["name"])
        except:
            return getattr(models, self.kw["model_kw"]["name"])

    def build_model(self):
        kw = self.kw.get("model_kw", {})
        lr = float(kw.get("lr", 1e-4))
        output_dir = self.output_uri
        make_dir(output_dir)
        backbone = self.model_type()(
            weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
            weights_backbone=ResNet50_Weights.DEFAULT,
            image_mean=[0.485, 0.456, 0.406],
            image_std=[0.229, 0.224, 0.225])
        num_classes = len(self.cc) + 1
        num_boxes = num_classes * 4
        backbone.roi_heads.box_predictor.cls_score = torch.nn.Linear(1024, num_classes)
        backbone.roi_heads.box_predictor.bbox_pred = torch.nn.Linear(1024, num_boxes)
        model = LightningModel(backbone, lr=lr)
        return model
    
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
        backbone = self.model_type()(num_classes=len(self.cc) + 1)
        backbone.load_state_dict(state_dict)
        model = LightningModel(backbone)
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
            check_val_every_n_epoch=1,
            callbacks=[LearningRateMonitor()]
        )
        train_dl, val_dl = self.build_train_loader(), self.build_val_loader()
        trainer.fit(model, train_dl, val_dl)
        trainer.save_checkpoint(output_dir + "/trainer/final-model.ckpt")

    def prediction_iterator(self, pred_dl, model, n=None):
        for i, (x, _) in tqdm(enumerate(pred_dl)):
            if n is not None and n != "None" and n > 0 and i >= n:
                break
            with torch.inference_mode():
                out_batch = model.predict(x)
                out_batch = output_to_array(out_batch, class_id_key="class_ids")
            for out in out_batch:
                yield out
                
    def predict(self):
        kw = self.kw.get("predict_kw", {})
        model = self.reconstruct_model()
        pred_dl = self.build_pred_loader()
        predictions = self.prediction_iterator(pred_dl, model, kw.get("max_batches", None))
        pred_labels = ObjectDetectionLabels.from_predictions(
            pred_dl.dataset.windows,
            predictions,
        )
        pred_labels = ObjectDetectionLabels.prune_duplicates(
            pred_labels,
            score_thresh=0.75,
            merge_thresh=0.1
        )
        pred_labels.save(
            uri=f"{self.output_uri}/pred-labels.geojson",
            crs_transformer=pred_dl.dataset.scene.raster_source.crs_transformer,
            class_config=self.cc,
        )
        return pred_labels
    
run_if_main(__name__, RVLightning)