# -*- coding: utf-8 -*-
"""
Functions to integrate your model with the DEEPaaS API.
It's usually good practice to keep this file minimal, only performing the interfacing
tasks. In this way you don't mix your true code with DEEPaaS code and everything is
more modular. That is, if you need to write the predict() function in api.py, you
would import your true predict function and call it from here (with some processing /
postprocessing in between if needed).
For example:

    import mycustomfile

    def predict(**kwargs):
        args = preprocess(kwargs)
        resp = mycustomfile.predict(args)
        resp = postprocess(resp)
        return resp

To start populating this file, take a look at the docs [1] and at a canonical exemplar
module [2].

[1]: https://docs.deep-hybrid-datacloud.eu/
[2]: https://github.com/deephdc/demo_app
"""

import base64
import json
import pkg_resources
import os
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from PIL import Image
from webargs import fields

import multi_plankton_separation.config as cfg
from multi_plankton_separation.misc import _catch_error
from multi_plankton_separation.utils import get_model_instance_segmentation


@_catch_error
def get_metadata():
    """
    DO NOT REMOVE - All modules should have a get_metadata() function
    with appropriate keys.
    """
    distros = list(pkg_resources.find_distributions(str(cfg.BASE_DIR), only=True))
    if len(distros) == 0:
        raise Exception("No package found.")
    pkg = distros[0]  # if several select first

    meta_fields = {
        "name": None,
        "version": None,
        "summary": None,
        "home-page": None,
        "author": None,
        "author-email": None,
        "license": None,
    }
    meta = {}
    for line in pkg.get_metadata_lines("PKG-INFO"):
        line_low = line.lower()  # to avoid inconsistency due to letter cases
        for k in meta_fields:
            if line_low.startswith(k + ":"):
                _, value = line.split(": ", 1)
                meta[k] = value

    return meta


# def get_train_args():
#     arg_dict = {
#         "epoch_num": fields.Int(
#             required=False,
#             missing=10,
#             description="Total number of training epochs",
#         ),
#     }
#     return arg_dict


# def train(**kwargs):
#     """
#     Dummy training. We just sleep for some number of epochs (1 epoch = 1 second)
#     mimicking some computation taking place.
#     We can log some random losses in Tensorboard to mimic monitoring.
#     """
#     logdir = BASE_DIR / "runs" / time.strftime("%Y-%m-%d_%H-%M-%S")
#     writer = SummaryWriter(logdir=logdir)
#     launch_tensorboard(logdir=logdir)
#     for epoch in range(kwargs["epoch_num"]):
#         time.sleep(1.)
#         writer.add_scalar("scalars/loss", - math.log(epoch + 1), epoch)
#     writer.close()

#     return {"status": "done", "final accuracy": 0.9}


def get_predict_args():
    """
    TODO: add more dtypes
    * int with choices
    * composed: list of strs, list of int
    """
    # WARNING: missing!=None has to go with required=False
    # fmt: off
    list_models = [filename[:-3] for filename in os.listdir(cfg.MODEL_DIR) if filename.endswith(".pt")]
    arg_dict = {
         "image": fields.Field(
             required=True,
             type="file",
             location="form",
             description="An image containing plankton to separate",
         ),
        "model": fields.Str(
            required=False,
            missing=list_models[0],
            enum = list_models,
            description = "The model used to perform instance segmentation"
        ),
        "threshold": fields.Float(
            required=False,
            missing=0.9,
            description="The minimum confidence score for a mask to be selected"
        ),
    }
    # fmt: on
    return arg_dict


@_catch_error
def predict(**kwargs):
    """
    Return same inputs as provided. We also add additional fields
    to test the functionality of the Gradio-based UI [1].
       [1]: https://github.com/deephdc/deepaas_ui
    """
    #kwargs = {"model": "mask_multi_plankton_b8", "threshold": 0.9}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model_path = '{}/{}.pt'.format(cfg.MODEL_DIR, kwargs["model"])
    cat_path = '{}/categories_{}.txt'.format(cfg.MODEL_DIR, kwargs["model"])

    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        model = get_model_instance_segmentation(list(state_dict["roi_heads.mask_predictor.mask_fcn_logits.bias"].size())[0])
        model.load_state_dict(state_dict)
        CATEGORIES = []
        with open(cat_path, 'r') as filehandle:
            for line in filehandle:
                currentPlace = line[:-1]
                CATEGORIES.append(currentPlace)
            
    else:
        message = 'Model not found.'
        return message
    
    # Convert image to tensor.
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()]) 
    orig_img = Image.open(kwargs['image'].filename)
    #orig_img = Image.open("/Users/emmaamblard/workarea/git/deep_project/img_00001.png")
    img = transform(orig_img)

    # Get predicted masks
    model.eval()
    pred = model([img])
    threshold= float(kwargs['threshold'])

    pred_class = [CATEGORIES[i] for i in list(pred[0]['labels'].numpy())] 
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())]
    pred_score = list(pred[0]['scores'].detach().numpy())
    pred_masks = (pred[0]['masks'].detach().numpy() > 0.7).squeeze(1)
    try:
        pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]
        pred_boxes = pred_boxes[:pred_t+1]   #Boxes.
        pred_class = pred_class[:pred_t+1]   #Name of the class.
        pred_score = pred_score[:pred_t+1]   #Prediction probability.
        pred_masks = pred_masks[:pred_t+1]
    except IndexError:
        pred_t = 'null'
        pred_boxes = 'null'
        pred_class = 'null'
        pred_score = 'null'
        pred_masks = 'null'
    
    fig, axes = plt.subplots(nrows=1, ncols=4, 
                            figsize=(3 * 4, 3), 
                            subplot_kw={'xticks': [], 'yticks': []})
    axes[0].imshow(orig_img, interpolation='none')
    pred_masks_hm = pred[0]["masks"].detach().numpy().squeeze(1)
    mask_sum = np.zeros(pred_masks[0].shape)
    mask_centers = []
    for i in range(len(pred_boxes)):
        mask_sum += pred_masks_hm[i]
        non_zero_x, non_zero_y = np.nonzero(pred_masks[i])
        mask_centers.append((int(non_zero_x.mean()), int(non_zero_y.mean())))
    axes[1].imshow(mask_sum, cmap="viridis")
    axes[1].set_title("Detected objects: {}".format(len(pred_boxes)))
    
    markers_mask = np.zeros(mask_sum.shape, dtype=bool)
    for (x, y) in mask_centers:
        markers_mask[x, y] = True
    markers, _ = ndi.label(markers_mask)
    watershed_mask = np.zeros(mask_sum.shape, dtype='int64')
    watershed_mask[mask_sum > 0.1] = 1
    labels = watershed(-mask_sum, markers, mask=watershed_mask, watershed_line=True)
    labels[watershed_mask == 0] = -1
    axes[2].imshow(labels, interpolation='none')

    lines = np.zeros(mask_sum.shape)
    lines[labels == 0] = 1
    x, y = np.nonzero(lines)
    axes[3].imshow(orig_img, interpolation='none')
    axes[3].scatter(y, x, s=1, color='red')

    result_path = os.path.join(cfg.TEMP_DIR, "pred_result.png")
    plt.savefig(result_path, bbox_inches='tight')
    plt.close()

    message = "Result saved in {}".format(result_path)

    return message


if __name__ == '__main__':
    message = predict()
    print(message)
    pass