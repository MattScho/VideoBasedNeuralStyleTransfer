'''
Manages segmenting images

:author: Matthew Schofield
'''
import torch
from instance_segmentation.yolact import Yolact
from instance_segmentation.utils.augmentations import FastBaseTransform
from instance_segmentation.layers.output_utils import postprocess
from instance_segmentation.data import cfg

class ImageSegmenter:

    def __init__(self, segmenter_weights):
        '''
        Initialize image segmenter with underlying YOLACT model
        '''
        self.model = Yolact()
        self.model.load_weights(segmenter_weights)
        self.model.eval()

    def segment(self, frame):
        '''
        Parse masks from frame using Yolact to identify instance segments

        :param frame: input image frame
        :return: List of masks for each instance in the frame
        '''
        # Init empty array to store masks
        output_masks = []
        # Prepare frame to be input to model
        frame = torch.from_numpy(frame).float()
        batch = FastBaseTransform()(frame.unsqueeze(0))
        # Evaluate batch
        preds = self.model(batch)
        # Parse masks from predictions
        masks = self.parse_masks(preds, frame)

        # Step through masks and convert to numpy arrays for output
        for mask in masks:
            output_masks.append(mask.detach().numpy())

        return output_masks

    def parse_masks(self, dets_out, img,):
        score_threshold = 0.15
        top_k = 3

        h, w, _ = img.shape

        save = cfg.rescore_bbox
        cfg.rescore_bbox = True
        t = postprocess(dets_out, w, h, visualize_lincomb=False,
                        crop_masks=True,
                        score_threshold=score_threshold)
        cfg.rescore_bbox = save

        idx = t[1].argsort(0, descending=True)[:top_k]

        if cfg.eval_mask_branch:
            # Masks are drawn on the GPU, so don't copy
            masks = t[3][idx]
        classes, scores, boxes = [x[idx].detach().numpy() for x in t[:3]]

        num_dets_to_consider = min(top_k, classes.shape[0])
        for j in range(num_dets_to_consider):
            if scores[j] < score_threshold:
                num_dets_to_consider = j
                break

        # First, draw the masks on the GPU where we can do it really fast
        # Beware: very fast but possibly unintelligible mask-drawing code ahead
        # I wish I had access to OpenGL or Vulkan but alas, I guess Pytorch tensor operations will have to suffice
        if cfg.eval_mask_branch and num_dets_to_consider > 0:
            # After this, mask is of size [num_dets, h, w, 1]
            masks = masks[:num_dets_to_consider, :, :, None]
            return masks
        return []
