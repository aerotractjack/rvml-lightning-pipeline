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
    
class RVLightningTraining:
    
    def __init__(self, tr_uris, val_uris, cc_names, cc_colors):
        self.train_uris = tr_uris
        self.val_uris = val_uris
        self.cc = ClassConfig(names=cc_names, colors=cc_colors, null_class="null")
        
    def build_train_ds(self):
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
            size_lims=(300, 350),
            out_size=325,
            max_windows=10,
            transform=data_augmentation_transform)
        return train_ds
    
    def build_val_ds(self):
        val_ds = SemanticSegmentationSlidingWindowGeoDataset.from_uris(
            class_config=self.cc,
            image_uri=self.val_uris[0],
            aoi_uri=self.val_uris[2],
            label_vector_uri=self.val_uris[1],
            label_vector_default_class_id=self.cc.get_class_id('DF'),
            size=325,
            stride=325)
        return val_ds
    
    def train(self):
        batch_size = 8
        lr = 1e-4
        epochs = 3
        output_dir = './semseg-trees-lightning/'
        make_dir(output_dir)
        fast_dev_run = False
        print(len(self.cc))
        print(self.cc)
        deeplab = deeplabv3_resnet50(num_classes=len(self.cc) + 1)
        model = SemanticSegmentation(deeplab, lr=lr)
        tb_logger = TensorBoardLogger(save_dir=output_dir + "tensorboard", flush_secs=10)
        trainer = pl.Trainer(
            accelerator='auto',
            min_epochs=1,
            max_epochs=epochs+1,
            default_root_dir=output_dir + "trainer/",
            logger=[tb_logger],
            fast_dev_run=fast_dev_run,
            log_every_n_steps=1,
        )
        tds = self.build_train_ds()
        vds = self.build_val_ds()
        train_dl = DataLoader(tds, batch_size=batch_size, shuffle=True, num_workers=4)
        val_dl = DataLoader(vds, batch_size=batch_size, num_workers=4)
        trainer.fit(model, train_dl, val_dl)
        trainer.save_checkpoint(output_dir + "trainer/final-model.ckpt")
        
    def prediction_iterator(self, pred_dl, model, n=None):
        for i, (x, _) in enumerate(pred_dl):
            if n is not None and i >= n:
                break
            with torch.inference_mode():
                out_batch = model(x)
            for out in out_batch:
                yield out.numpy()
        
    def predict(self, n=10):
        def fix_keys(data):
            for _ in range(len(data)):
                k, v = data.popitem(False)
                newk = k[k.index(".")+1:]
                data[newk] = v
            return data
        ckpt_path = './semseg-trees-lightning/trainer/final-model.ckpt'
        ckpt = torch.load(ckpt_path)
        state_dict = ckpt["state_dict"]
        state_dict = fix_keys(state_dict)
        print(self.cc)
        deeplab = deeplabv3_resnet50(num_classes=len(self.cc) + 1)
        deeplab.load_state_dict(state_dict)
        model = SemanticSegmentation(deeplab)
        pds = self.build_val_ds()
        pred_dl = DataLoader(pds, batch_size=8, num_workers=4)
        predictions = self.prediction_iterator(pred_dl, model, n)
        pred_labels = SemanticSegmentationLabels.from_predictions(
            pds.windows,
            predictions,
            smooth=True,
            extent=pds.scene.extent,
            num_classes=len(self.cc)+1
        )
        scores = pred_labels.get_score_arr(pred_labels.extent)
        return pred_labels, scores
        
if __name__ == "__main__":
    train_image_uri = "/home/aerotract/NAS/main/Clients/Giustina/Winter2023/100_230OG/Data/ortho/230OG_Orthomosaic_export_WedMar08181810258552.tif"
    train_label_uri = "/home/aerotract/NAS/main/Clients/Giustina/Winter2023/100_230OG/Training/Tree_polygons_00_230OG.geojson"
    train_aoi_uri = "/home/aerotract/NAS/main/Clients/Giustina/Winter2023/100_230OG/Training/border_polygon_00_230OG.geojson"

    val_image_uri = "/home/aerotract/NAS/main/Clients/Giustina/Winter2023/100_230OG/Data/ortho/230OG_Orthomosaic_export_WedMar08181810258552.tif"
    val_label_uri = "/home/aerotract/NAS/main/Clients/Giustina/Winter2023/100_230OG/Training/Tree_polygons_00_230OG.geojson"
    val_aoi_uri = "/home/aerotract/NAS/main/Clients/Giustina/Winter2023/100_230OG/Training/border_polygon_00_230OG.geojson"

    obj = RVLightningTraining(
        [train_image_uri, train_label_uri, train_aoi_uri],
        [val_image_uri, val_label_uri, val_aoi_uri],
        ["DF", "null"],
        ["orange", "black"]
    )

    obj.train()

    labels, scores = obj.predict()