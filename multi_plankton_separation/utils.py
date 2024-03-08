import os
import torch
import torchvision
import numpy as np

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation , MaskFormerImageProcessor
from scipy import ndimage as ndi
from skimage.segmentation import watershed, find_boundaries

import multi_plankton_separation.config as cfg
from PIL import Image


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


def load_saved_model_pano(model_name,device):
    """
    Load a panoptic saved model for a given model_name
    """
    import zipfile
    
    model_path = '{}/{}'.format(cfg.MODEL_DIR, model_name)

    if not os.path.exists(model_path[:-4]):
        with zipfile.ZipFile(model_path, 'r') as zip_ref:
            zip_ref.extractall(model_path.strip(model_path.split("/")[-1]))    
    
    model_path=model_path[:-4]

    if not os.path.exists(model_path):
        print("Model {} not found.".format(model_name))
        return None
    
    processor = MaskFormerImageProcessor.from_pretrained(model_path)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(model_path)
    model.to(device)
    
    return model,processor


def predict_mask_maskrcnn(model, image_path, score_threshold, mask_threshold):
    """
    Perform the mask segmentation for a given image with a maskRCNN model
    """
    # Convert image to tensor
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    orig_img = Image.open(image_path)
    img = transform(orig_img)

    # Get predicted masks
    pred_masks, pred_masks_probs = get_predicted_masks(
        model, img, score_threshold
    )

    # Get sum of masks probabilities and mask centers
    mask_sum = np.zeros(pred_masks[0].shape)
    mask_centers_x = []
    mask_centers_y = []

    # Get sum of masks and mask centers for the watershed
    for mask in pred_masks_probs:
        to_add = mask
        to_add[to_add < mask_threshold] = 0
        mask_sum += to_add
        center_x, center_y = np.unravel_index(np.argmax(mask), mask.shape)
        mask_centers_x.append(center_x)
        mask_centers_y.append(center_y)
    mask_centers = zip(mask_centers_x, mask_centers_y)

    # Get silhouette of objects to use as a mask for the watershed
    binary_img = (img[0, :, :] + img[1, :, :] + img[2, :, :] != 3).numpy().astype(float)
    
    mask_centers = zip(mask_centers_x, mask_centers_y)
    
    return mask_sum, mask_centers_x, mask_centers_y, binary_img, pred_masks_probs, len(pred_masks)


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


def predict_mask_panoptic(model, processor, image_path, device, score_threshold=0.9):
    """
    Perform the mask segmention for a given image with a panoptic model
    """
    import albumentations as A
    
    image = Image.open(image_path).convert("RGB")
    
    ADE_MEAN = np.array([123.675, 116.280, 103.530]) / 255
    ADE_STD = np.array([58.395, 57.120, 57.375]) / 255

    image_transform = A.Compose([
        A.Resize(width=512, height=512),
        A.Normalize(mean=ADE_MEAN, std=ADE_STD),
    ])
    # Get network input
    pixel_values = image_transform(image=np.array(image))["image"]
    pixel_values = np.moveaxis(pixel_values, -1, 0)
    pixel_values = torch.from_numpy(pixel_values).unsqueeze(0)
    
    # Predict and get panoptic masks
    with torch.no_grad():
        outputs = model(pixel_values.to(device))
    results = processor.post_process_panoptic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
    panoptic_masks = results["segmentation"].cpu().numpy()
    
    # Get binary image separating the background (0) from objects (1)
    gray_img = np.array(image.convert('L'))
    binary_image = (gray_img < 255).astype(float)
    
    # Compute distances and mask centers
    distances = np.zeros(panoptic_masks.shape)
    mask_centers = list()
    scores = list()
    for segment_info in results["segments_info"]:
        if segment_info["score"] < score_threshold:
            continue          
        if segment_info["label_id"] == 0:
            continue
        else:
            single_mask = (panoptic_masks == segment_info["id"]).astype(int)
            
        dist = ndi.distance_transform_edt(single_mask)
        distances += dist
        ind = np.unravel_index(np.argmax(dist, axis=None), dist.shape)
        mask_centers.append((ind[0], ind[1]))
        scores.append(segment_info["score"])
    
    return distances,mask_centers,binary_image



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
