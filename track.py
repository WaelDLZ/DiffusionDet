import argparse
import PIL
import cv2
import os
import torch
from detectron2.modeling import build_model
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
import detectron2.data.transforms as T
from detectron2.utils.visualizer import Visualizer
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import MetadataCatalog



from diffusiondet.util.model_ema import add_model_ema_configs
from diffusiondet import add_diffusiondet_config



def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 tracking for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/diffdet.coco.res50.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--input",
        help="A directory path with video frames; ",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser

def track(args,cfg):

    aug = T.ResizeShortestEdge([cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST)
    

    model = build_model(cfg)
    model.eval()

    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)

    metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused")
    
    for img_name in sorted(os.listdir(args.input)):
        print(img_name)
        path=os.path.join(args.input,img_name)
        original_image=read_image(path,format="BGR")
        with torch.no_grad(): 
            # Apply pre-processing to image.
        
            if cfg.INPUT.FORMAT == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = aug.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

            inputs = {"image": image, "height": height, "width": width}
            predictions = model([inputs])[0]
            instances = predictions['instances']
            instances = instances[instances.scores > args.confidence_threshold]
            print(len(instances)," instances detected")
            visualizer = Visualizer(original_image,metadata)
            if args.output and os.path.isdir(args.output):
                out_filename = os.path.join(args.output, img_name)
                vis_output = visualizer.draw_instance_predictions(predictions=instances)
                vis_output.save(out_filename)
        
if __name__=="__main__":
    args = get_parser().parse_args()
    cfg = get_cfg()
    add_diffusiondet_config(cfg)
    add_model_ema_configs(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    track(args,cfg)
