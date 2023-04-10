from rastervision.pytorch_learner.object_detection_utils import TorchVisionODAdapter
import torch
import pytorch_lightning as pl
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from util import output_to_array

class LightningModel(pl.LightningModule):

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
    
    def training_step(self, batch, batch_ind):
        x, y = batch
        loss_dict = self.backbone(x, y)
        loss_dict['loss'] = sum(loss_dict.values())
        self.log("train_loss", loss_dict["loss"])
        return loss_dict

    def validation_step(self, batch, batch_ind):
        x, ys = batch
        outs = self.backbone(x)
        ys = output_to_array(ys, class_id_key="labels")
        outs = output_to_array(outs, class_id_key="labels")
        self.val_map_metric.update(outs, ys)
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
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=10
        )
        return {"optimizer": optimizer, "lr_scheduler": sched, "monitor": "train_loss"}