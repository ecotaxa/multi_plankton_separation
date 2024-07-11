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

import pkg_resources
import os
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from PIL import Image
from webargs import fields, validate

import multi_plankton_separation.config as cfg
from multi_plankton_separation.misc import _catch_error
from multi_plankton_separation.utils import (
    load_saved_model,
    load_saved_model_pano,
    predict_mask_maskrcnn,
    predict_mask_panoptic,
    get_watershed_result,
    bounding_box
)


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


def get_predict_args():
    """
    Get the list of arguments for the predict function
    """
    # Get list of available models
    
    
    #######Mettre à jour la liste des modèles
    list_models = list()

    for filename in os.listdir(cfg.MODEL_DIR):
        if filename.endswith(".pt"):
            list_models.append(filename[:-3])
        elif "pano" in filename:
            list_models.append(filename)
                        

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
            enum=list_models,
            description="The model used to perform instance segmentation"
        ),
        "min_mask_score": fields.Float(
            required=False,
            missing=0.9,
            description="The minimum confidence score for a mask to be selected"
        ),
        "min_mask_value": fields.Float(
            required=False,
            missing=0.5,
            description="The minimum value for a pixel to belong to a mask"
        ),
        "accept" : fields.Str(
            required=False,
            missing='image/png',
            validate=validate.OneOf(['image/png']),
            description="Return an image or a json with the path to the saved result"
        ),
    }

    return arg_dict


@_catch_error
def predict(**kwargs):
    """
    Prediction function
    """
    
    # Check if a GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    if 'default' in kwargs["model"]:
        model_type='rcnn'
    else:
        model_type='panoptic'
        
    #model_type='panoptic'
    if model_type == 'rcnn':
        model = load_saved_model(kwargs["model"], device)
    else:
        model, processor = load_saved_model_pano(kwargs["model"], device)
        
    if model is None:
        message = "Model not found."
        return message


    # Get predicted masks
    score=0
    if model_type == 'rcnn':
        mask_sum, centersx, centersy, binary_img, pred_masks_probs, nb_obj_detected = predict_mask_maskrcnn(model, kwargs['image'].filename, kwargs["min_mask_score"], kwargs["min_mask_value"])
        mask_centers = zip(centersx,centersy)
    else:
        mask_sum, mask_centers, binary_img, score = predict_mask_panoptic(model, processor, kwargs['image'].filename, device, kwargs["min_mask_score"])
     
    # Apply watershed algorithm
    watershed_labels = get_watershed_result(mask_sum, mask_centers, mask=binary_img)

    # Save output separations
    separation_mask = np.ones(watershed_labels.shape)
    separation_mask[watershed_labels != 0] = 'nan'
    lines_image = Image.fromarray(separation_mask * 255).convert('L')
    output_path = os.path.join(cfg.TEMP_DIR, "out_image.png")
    lines_image.save(output_path)

    plot_width = mask_sum.shape[0] + 1000
    plot_height = mask_sum.shape[1] + 1000
    px = 1 / plt.rcParams['figure.dpi']
    fig, axes = plt.subplots(nrows=1, ncols=5,
                             figsize=(plot_width * 5 * px, plot_height * px),
                             subplot_kw={'xticks': [], 'yticks': []})

    # Plot original image
    orig_img = Image.open(kwargs['image'].filename)
    axes[0].imshow(orig_img, interpolation='none')
    if model_type=='rcnn':
        for mask in pred_masks_probs:
            rmin, rmax, cmin, cmax = bounding_box(mask)
            x, y = cmin, rmin
            width, height = cmax - cmin, rmax - rmin
            rect = patches.Rectangle((x, y), width, height,
                                     linewidth=1, edgecolor='r', facecolor='none')
            axes[0].add_patch(rect)
        axes[0].set_title("Detected objects: {}".format(nb_obj_detected))

    # Plot mask map
    axes[1].imshow(mask_sum, cmap="viridis")
    axes[1].set_title("Sum of predicted masks")

    # Plot watershed results
    axes[2].imshow(watershed_labels, interpolation='none')
    
    if model_type=='rcnn':
        axes[2].scatter(centersx, centersy, color='red')
    axes[2].set_title("Watershed with markers")

    # Plot original image with separations
    axes[3].imshow(orig_img, interpolation='none')
    axes[3].imshow(separation_mask, interpolation='none')
    axes[3].set_title("Extracted line(s)")

    # Plot output
    axes[4].imshow(lines_image, cmap='Greys_r', interpolation='none')
    axes[4].set_title("Output")

    # Save plot
    result_path = os.path.join(cfg.TEMP_DIR, "pred_result.png")
    plt.savefig(result_path, bbox_inches='tight')
    plt.close()

    if kwargs["accept"] == 'image/png':
        message = open(output_path, 'rb')
    else:
        message = "Result saved in {}".format(output_path)

    return message, str(score)

