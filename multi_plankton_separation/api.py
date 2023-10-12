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
import cv2

from PIL import Image
from webargs import fields

import multi_plankton_separation.config as cfg
from multi_plankton_separation.misc import _catch_error
from multi_plankton_separation.utils import (
    load_saved_model,
    get_predicted_masks,
    get_watershed_result
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
    Get the list of arguments for the predict function
    """
    # Get list of available models
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

    return arg_dict


@_catch_error
def predict(**kwargs):
    """
    Prediction function
    """
    #kwargs = {"model": "mask_multi_plankton_b8", "threshold": 0.9}

    # Check if a GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = load_saved_model(kwargs["model"], device)
    if model is None:
        message = "Model not found."
        return message
    
    # Convert image to tensor
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()]) 
    orig_img = Image.open(kwargs['image'].filename)
    #orig_img = Image.open("/Users/emmaamblard/workarea/git/deep_project/img_00001.png")
    img = transform(orig_img)

    # Get predicted masks
    pred_masks, pred_masks_probs = get_predicted_masks(model, img, kwargs["threshold"])
    
    # Get sum of masks probabilities and mask centers
    mask_sum = np.zeros(pred_masks[0].shape)
    mask_centers = []

    for i in range(len(pred_masks_probs)):
        mask_sum += pred_masks_probs[i]
        non_zero_x, non_zero_y = np.nonzero(pred_masks[i])
        mask_centers.append((int(non_zero_x.mean()), int(non_zero_y.mean())))

    # Apply watershed algorithm
    labels = get_watershed_result(mask_sum, mask_centers)

    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(3 * 4, 3), subplot_kw={'xticks': [], 'yticks': []})

    # Plot original image
    axes[0].imshow(orig_img, interpolation='none')

    # Plot mask map
    axes[1].imshow(mask_sum, cmap="viridis")
    axes[1].set_title("Detected objects: {}".format(len(pred_masks)))

    # Plot watershed results
    axes[2].imshow(labels, interpolation='none')

    # Plot original image with separations
    axes[3].imshow(orig_img, interpolation='none')
    separation_mask = np.ones(labels.shape)
    separation_mask[labels != 0] = 'nan'
    axes[3].imshow(separation_mask, cmap="Greys", interpolation='none')

    # Save plot
    result_path = os.path.join(cfg.TEMP_DIR, "pred_result.png")
    plt.savefig(result_path, bbox_inches='tight')
    plt.close()

    # Get output image with separations
    cv2_img = cv2.cvtColor(np.array(orig_img), cv2.COLOR_BGR2GRAY)
    cv2_img[separation_mask == 1] = 255
    output_path = os.path.join(cfg.TEMP_DIR, "out_image.png")
    cv2.imwrite(output_path, cv2_img)

    message = "Result saved in {}".format(result_path)

    return message


if __name__ == '__main__':
    message = predict()
    print(message)
    pass