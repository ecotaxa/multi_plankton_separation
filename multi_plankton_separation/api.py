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
from skimage.segmentation import find_boundaries

import multi_plankton_separation.config as cfg
from multi_plankton_separation.misc import _catch_error
from multi_plankton_separation.utils import (
    load_saved_model,
    get_predicted_masks,
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
        "accept" : fields.Str(
            required=False,
            missing='image/png',
            validate=validate.OneOf(['image/png']),
            description="Returns an image or a json with the path to the saved result"),
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
    #orig_img = Image.open("/Users/emmaamblard/Downloads/seg_data/images/img_00105.png")
    img = transform(orig_img)

    # Get predicted masks
    pred_masks, pred_masks_probs = get_predicted_masks(model, img, kwargs["threshold"])
    
    # Get sum of masks probabilities and mask centers
    mask_sum = np.zeros(pred_masks[0].shape)
    mask_centers_x = []
    mask_centers_y = []

    for mask in pred_masks_probs:
        mask_sum += mask
        center_x, center_y = np.unravel_index(np.argmax(mask), mask.shape)
        mask_centers_x.append(center_x)
        mask_centers_y.append(center_y)
    
    mask_centers = zip(mask_centers_x, mask_centers_y)

    # Apply watershed algorithm
    watershed_labels = get_watershed_result(mask_sum, mask_centers)

    # Save output separations
    separation_mask = np.ones(watershed_labels.shape)
    separation_mask[watershed_labels != 0] = 'nan'
    lines_image = Image.fromarray(separation_mask * 255).convert('L')
    output_path = os.path.join(cfg.TEMP_DIR, "out_image.png")
    lines_image.save(output_path)

    plot_width = mask_sum.shape[0] + 1000
    plot_height = mask_sum.shape[1] + 1000
    px = 1/plt.rcParams['figure.dpi']
    fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(plot_width * 5 * px, plot_height * px), subplot_kw={'xticks': [], 'yticks': []})

    # Plot original image
    axes[0].imshow(orig_img, interpolation='none')
    for mask in pred_masks_probs:
        rmin, rmax, cmin, cmax = bounding_box(mask)
        x, y = cmin, rmin
        width, height = cmax - cmin, rmax - rmin
        rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor='r', facecolor='none')
        axes[0].add_patch(rect)
    axes[0].set_title("Detected objects: {}".format(len(pred_masks)))

    # Plot mask map
    axes[1].imshow(mask_sum, cmap="viridis")
    axes[1].set_title("Sum of predicted masks")

    # Plot watershed results
    axes[2].imshow(watershed_labels, interpolation='none')
    axes[2].scatter(mask_centers_y, mask_centers_x, color='red')
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

    if(kwargs["accept"] == 'image/png'):
        message = open(output_path, 'rb')
    else:
        message = "Result saved in {}".format(output_path)

    return message


if __name__ == '__main__':
    message = predict()
    print(message)
    pass