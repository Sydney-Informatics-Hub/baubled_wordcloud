#!/usr/bin/env python
# coding: utf-8
"""Fill an image with a word cloud and circle pictures

See generate_word_cloud_with_baubles below.

"""
# Copyright 2019 The University of Sydney
#
# MIT License. See COPYING
#
# Created by Joel Nothman at the Sydney Informatics Hub
#
#       ___                       ___
#      /  /\        ___          /__/\
#     /  /:/_      /  /\         \  \:\
#    /  /:/ /\    /  /:/          \__\:\
#   /  /:/ /::\  /__/::\      ___ /  /::\
#  /__/:/ /:/\:\ \__\/\:\__  /__/\  /:/\:\
# \  \:\/:/~/:/    \  \:\/\ \  \:\/:/__\/
#   \  \::/ /:/      \__\::/  \  \::/
#    \__\/ /:/       /__/:/    \  \:\
#      /__/:/ please \__\/      \  \:\
#      \__\/ acknowledge your use\__\/

from PIL import Image, ImageDraw
import numpy as np
import wordcloud
from matplotlib import colors
from tqdm import tqdm
import random


# XXX: With apologies, this overuses np.asarray(image).
#      Would be nice to clean this up.


def make_circle_mask(orig_arr, cx, cy, r):
    """Create a boolean array with a circle where it is True

    Parameters
    ----------
    orig_arr : 2d array
        Defines the output shape.
    cx, cy : int
        The coordinate of the centre of the circle
    r : int
        Radius of the circle

    Returns
    -------
    2d bool array
        Only cells within the circle centred at (cx, cy) with radius r have
        value True.
    """
    height = orig_arr.shape[0]
    width = orig_arr.shape[1]
    # array of distance from centre
    y_dist = np.arange(height).reshape(-1, 1) - cy
    x_dist = np.arange(width).reshape(1, -1) - cx
    return x_dist ** 2 + y_dist ** 2 <= r ** 2


def get_white_mask(image):
    # image is array or Image
    return (np.asarray(image)[:, :, :3] == 255).all(axis=-1)


def place_baubles(
    mask_image_arr, n, min_radius, max_radius, margin=3, min_sep=3, seed=0
):
    """Selects locations of baubles and masks them out from mask_image_arr

    Parameters
    ----------
    mask_image_arr : array
        RGB image. Non-white spaces to be filled with baubles.
    n : int
        Number of baubles
    min_radius : int
    max_radius : int
    margin : int
        Gap beyond radius to leave masked
    min_sep : int
        Distance requried between baubles.
    seed : int
        Random seed

    Returns
    -------
    locations : list of tuples
        Tuples are (y, x, r) triples
    mask_image_arr : array
        RGB image. Compared to input, baubles are now white.
    """
    mask_image_arr = mask_image_arr.copy()

    locations = []
    rng = np.random.RandomState(seed)

    for i in tqdm(range(n), desc='Placing baubles'):
        bool_mask = ~get_white_mask(mask_image_arr)
        # can't place a circle within radius of a masked spot
        blur = int((min_radius + margin) * (2 ** .5)) + margin
        blur_mask = bool_mask.copy()
        blur_mask[blur:] &= bool_mask[:-blur]
        blur_mask[:-blur] &= bool_mask[blur:]
        blur_mask[:, blur:] &= bool_mask[:, :-blur]
        blur_mask[:, :-blur] &= bool_mask[:, blur:]
        Y, X = blur_mask.nonzero()

        ys = np.arange(len(Y))
        rng.shuffle(ys)
        for i in ys:
            y = Y[i]
            x = X[i]
            r = rng.randint(min_radius, max_radius)
            if (
                y + r > mask_image_arr.shape[0]
                or y - r < 0
                or x + r > mask_image_arr.shape[1]
                or x - r < 0
            ):
                continue
            safe_mask = make_circle_mask(mask_image_arr, x, y, r + min_sep + margin)
            circle_mask = make_circle_mask(mask_image_arr, x, y, r + margin)
            if bool_mask[safe_mask].all():
                # okay if not overlapping
                break
        else:
            raise RuntimeError(
                "Could not place all baubles. "
                "Use fewer or smaller baubles, or try a different random seed."
            )
        locations.append((y, x, r))
        mask_image_arr[circle_mask] = 255

    return locations, mask_image_arr


def calc_weights_from_lengths(words, length_exponent=0.8, seed=0):
    """Weight words randomly with a sublinear preference for length

    Parameters
    ----------
    words : list of str
    length_exponent : float
        The length of the word is raised to this value before multiplying by
        a uniform random 0-1 weight.
    seed : int
        Random seed

    Returns
    -------
    dict
        Mapping from word to weight
    """
    return dict(
        zip(
            words,
            np.random.RandomState(seed).rand(len(words))
            * [len(word) ** length_exponent for word in words],
        )
    )


def clear_unmasked(image, mask_image, bg_color="white"):
    """Whiten the outside of the mask in image
    """
    inverted_mask = get_white_mask(mask_image)
    arr = np.asarray(image).copy()
    arr[inverted_mask] = list(np.asarray(colors.to_rgb(bg_color)) * 255)
    return Image.fromarray(arr)


def crop_image_to_circle(image):
    """Crops an image to the topmost circle

    Outside the circle is set to be transparent

    Parameters
    ----------
    image : Image

    Returns
    -------
    Image
        Square image in RGBA
    """
    arr = np.asarray(image.convert("RGB"))
    arr = arr[: arr.shape[1]]  # crop to square
    x = y = arr.shape[1] // 2
    # construct a mask that can be used to index the arr
    # XXX: there should be a nicer way to code this using broadcasting.
    circle_mask = np.moveaxis(
        np.repeat([~make_circle_mask(arr, x, y, x)], arr.shape[-1], axis=0),
        source=0,
        destination=-1,
    )
    # make the outside of the circle transparent
    rgba_shape = (arr.shape[0], arr.shape[1], 4)
    arr_rgba = np.full(rgba_shape, 255).astype("uint8")
    arr_rgba[:, :, :-1] = arr
    arr_rgba[:, :, -1][circle_mask[:, :, 0]] = 0
    return Image.fromarray(arr_rgba)


def paste_bauble_images(image, bauble_paths, locations, outline=None):
    """Cut images to circles and paste at given locations on image

    Parameters
    ----------
    image : Image
    bauble_paths : list of str
    locations : list of tuples
        Tuples are (y, x, r) triples as returned from place_baubles
    outline : str
        Colour of circle outline
    """
    draw = ImageDraw.Draw(image)
    for path, (y, x, r) in tqdm(zip(bauble_paths, locations), desc="Pasting images"):
        bauble_image = Image.open(path)
        bauble_image = crop_image_to_circle(bauble_image)
        bauble_image = bauble_image.resize((r * 2, r * 2))
        image.paste(bauble_image, (x - r, y - r), bauble_image)
        if outline:
            draw.ellipse((x - r, y - r, x + r, y + r), fill=None, outline=outline)


def generate_word_cloud_with_baubles(
    fill_image_path,
    bauble_pic_paths,
    word_freqs,
    min_bauble_radius=0.03,
    max_bauble_radius_ratio=1.5,
    bauble_margin=1,
    bauble_min_sep=3,
    cloud_bg_color="green",
    cloud_fg_colormap="Pastel1",
    image_bg_color="white",
    bauble_outline=False,
    seed=0,
):
    """Fill a mask with a word cloud and baubles (circular pictures)

    Parameters
    ----------
    fill_image_path : str
        Path of image where non-white spaces should be filled
    bauble_pic_paths : list of str
        Paths to images to paste in baubles
    word_freqs : dict
        Mapping of term to weigting for cloud.
    min_bauble_radius : int or float
        Minimum radius of a bauble.
        float < 1 indicates a fraction of the width of the whole image.
    max_bauble_radius_ratio : float
        Ratio between minimum and maximum bauble radius.
    bauble_margin : int
        Number of pixels around each bauble where words can't be placed.
    bauble_min_sep : int
        Number of pixels required to be kept between each bauble.
    cloud_bg_color : str
        Colour supported by PIL for the background of the word cloud, i.e.
        where the initial fill_image mask is.
    cloud_fg_colormap : str or colormap
        Matplotlib colormap for colouring words.
    image_bg_color : str
        Colour for the background of the image.
    bauble_outline : str or False
        Colour to outline baubles
    seed : int
        Random seed

    Returns
    -------
    Image
    """
    rng = random.Random(seed)

    def _gen_seed():
        return rng.randint(0, np.iinfo(np.int32).max)

    mask_image = Image.open(fill_image_path)

    if min_bauble_radius < 1:
        min_bauble_radius = int(np.ceil(mask_image.width * min_bauble_radius))
    max_bauble_radius = int(np.ceil(min_bauble_radius * max_bauble_radius_ratio))

    bauble_centres_and_radii, mask_image_arr = place_baubles(
        np.asarray(mask_image),
        n=len(bauble_pic_paths),
        min_radius=min_bauble_radius,
        max_radius=max_bauble_radius,
        margin=bauble_margin,
        min_sep=bauble_min_sep,
        seed=_gen_seed(),
    )

    if "," in cloud_fg_colormap:
        cloud_fg_colormap = colors.LinearSegmentedColormap.from_list(
            "my_list", [colors.to_rgba(s) for s in cloud_fg_colormap.split(",")], N=10
        )

    cloud = wordcloud.WordCloud(
        background_color=cloud_bg_color,
        repeat=True,
        mask=mask_image_arr,
        colormap=cloud_fg_colormap,
        random_state=_gen_seed(),
    )
    cloud.generate_from_frequencies(word_freqs)
    out = clear_unmasked(cloud.to_image(), mask_image, bg_color=image_bg_color)

    bauble_pic_paths = bauble_pic_paths[:]  # copy
    random.Random(_gen_seed()).shuffle(bauble_pic_paths)
    paste_bauble_images(
        out, bauble_pic_paths, bauble_centres_and_radii, outline=bauble_outline
    )
    return out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("bauble_paths", nargs="+")
    parser.add_argument(
        "-m",
        "--mask-image",
        default="xmas-tree.png",
        help="path to mask image. 'xmas-tree.png' by default.",
    )
    parser.add_argument(
        "-w",
        "--word-list",
        default="words-for-cloud.txt",
        help="path to word list. 'words-for-cloud.txt' by default.",
    )
    parser.add_argument("-s", "--seed", default=0, type=int, help="random seed")
    parser.add_argument(
        "-o",
        "--out-path",
        default="out.png",
        help="output image path, 'out.png' by default.",
    )
    parser.add_argument(
        "--bg-color", default="green", help="Color for cloud background",
    )
    parser.add_argument(
        "--image-bg-color", default="white", help="Color for image background",
    )
    parser.add_argument(
        "--word-colormap",
        default="Pastel1",
        help=(
            "Colormap for cloud words."
            " Specify gradients by a list of ,-separated color names."
        ),
    )
    parser.add_argument(
        "--min-bauble-radius",
        type=float,
        help=(
            "Minimum radius of a bauble."
            " <1 indicates a fraction of the width of the whole image."
        ),
    )
    args = parser.parse_args()

    words = [word.strip() for word in open(args.word_list)]
    out = generate_word_cloud_with_baubles(
        args.mask_image,
        bauble_pic_paths=args.bauble_paths,
        word_freqs=calc_weights_from_lengths(
            words, length_exponent=0.1, seed=args.seed
        ),
        cloud_bg_color=args.bg_color,
        image_bg_color=args.image_bg_color,
        cloud_fg_colormap=args.word_colormap,
        seed=args.seed,
        min_bauble_radius=args.min_bauble_radius,
    )
    out.save(args.out_path)
