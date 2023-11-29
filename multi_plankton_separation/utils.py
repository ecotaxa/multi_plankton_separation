import os
import torch
import torchvision
import numpy as np

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from scipy import ndimage as ndi
from skimage.segmentation import watershed, find_boundaries

import multi_plankton_separation.config as cfg


def get_model_instance_segmentation(num_classes):
    """
    Get a new instance segmentation model
    """
    # Load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # Initialize classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Initialize mask predictor
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )

    return model


def load_saved_model(model_name, device):
    """
    Load a saved model for a given model_name
    """
    model_path = '{}/{}.pt'.format(cfg.MODEL_DIR, model_name)

    if not os.path.exists(model_path):
        print("Model {} not found.".format(model_name))
        return None

    state_dict = torch.load(model_path, map_location=device)
    model = get_model_instance_segmentation(
        list(state_dict["roi_heads.mask_predictor.mask_fcn_logits.bias"].size())[0]
    )
    model.load_state_dict(state_dict)

    return model


def get_predicted_masks(model, image, score_threshold=0.9, mask_threshold=0.7):
    """
    Perform instance segmentation for a given image with the given model
    """
    model.eval()
    pred = model([image])

    pred_score = list(pred[0]['scores'].detach().numpy())
    pred_masks = (pred[0]['masks'].detach().numpy() > mask_threshold).squeeze(1)
    pred_masks_probs = pred[0]['masks'].detach().numpy().squeeze(1)
    try:
        pred_t = [pred_score.index(x) for x in pred_score if x > score_threshold][-1]
        pred_score = pred_score[:pred_t + 1]
        pred_masks = pred_masks[:pred_t + 1]
        pred_masks_probs = pred_masks_probs[:pred_t + 1]
    except IndexError:
        pred_t = 'null'
        pred_score = 'null'
        pred_masks = 'null'
        pred_masks_probs = 'null'

    return pred_masks, pred_masks_probs


def get_watershed_result(mask_map, mask_centers, mask=None):
    """
    Apply the watershed algorithm on the predicted mask map,
    using the mask centers as markers
    """
    # Prepare watershed markers
    markers_mask = np.zeros(mask_map.shape, dtype=bool)
    for (x, y) in mask_centers:
        markers_mask[x, y] = True
    markers, _ = ndi.label(markers_mask)

    # Prepare watershed mask
    if mask is None:
        watershed_mask = np.zeros(mask_map.shape, dtype='int64')
        watershed_mask[mask_map > .01] = 1
    else:
        watershed_mask = mask

    # Apply watershed
    labels = watershed(
        -mask_map, markers, mask=watershed_mask, watershed_line=False
    )

    # Derive separation lines
    lines = np.zeros(labels.shape)
    unique_labels = list(np.unique(labels))
    unique_labels.remove(0)

    for value in unique_labels:
        single_shape = (labels == value).astype(int)
        boundaries = find_boundaries(
            single_shape, connectivity=2, mode='outer', background=0
        )
        boundaries[(labels == 0) | (labels == value)] = 0
        lines[boundaries == 1] = 1

    labels_with_lines = labels
    labels_with_lines[labels_with_lines == 0] = -1
    labels_with_lines[lines == 1] = 0

    return labels_with_lines


def bounding_box(img):
    """
    Get the corners of the bounding box of an object on a white background
    """
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax
