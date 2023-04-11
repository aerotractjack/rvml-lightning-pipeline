import torch
from rastervision.pytorch_learner.object_detection_utils import BoxList
import sys

def exit():
    sys.exit(0)

def boxlist_to_tensor(boxlist, class_id_key):
    tns = {}
    tns["boxes"] = boxlist.convert_boxes('xyxy').type(torch.float)
    tns[class_id_key] = boxlist.get_field('class_ids').type(torch.int64) 
    scores = boxlist.get_field('scores')
    if scores is not None:
        tns["scores"] = scores.type(torch.float)
    return tns

def boxlist_to_numpy(boxlist, class_id_key):
    npy = {}
    npy["boxes"] = boxlist.convert_boxes('xyxy').cpu().numpy()
    npy[class_id_key] = boxlist.get_field('class_ids').cpu().numpy() 
    scores = boxlist.get_field('scores')
    if scores is not None:
        npy["scores"] = scores.cpu().numpy()
    return npy

def output_to_array(out, class_id_key="class_ids"):
    fnmap = {
        "class_ids": boxlist_to_numpy,
        "labels": boxlist_to_tensor
    }
    fn = fnmap[class_id_key]
    if isinstance(out, BoxList):
        return fn(out, class_id_key)
    else:
        return [fn(boxlist, class_id_key) for boxlist in out]
    
def dataloader_iterator(data, out_type="tensor"):
    tmap = {
        "numpy": "class_ids", 
        "tensor": "labels"
    }
    for x, y in data:
        y = output_to_array(y, class_id_key=tmap[out_type])
        yield (x, y)

def dict_to_tensor(xs):
    tens = None
    for x in xs:
        box = x["boxes"]
        if tens is None:
            tens = box
            continue
        if box.shape[0] == 0:
            continue
        tens = torch.cat((tens, box))
    return tens
    
def run_if_main(name, objcls):
    if name != "__main__":
        return
    from configreader import yaml2dict
    import sys
    file = "input/giustina230OG-lightning.yaml"
    if len(sys.argv) > 1:
        file = sys.argv[1]
    conf = yaml2dict(file)
    obj = objcls(
        conf["train_uri"],
        conf["val_uri"],
        conf["pred_uri"],
        conf["output"],
        conf["class_config"],
        conf
    )
    obj.run()