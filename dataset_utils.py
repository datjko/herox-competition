import json
import math
import numpy as np
import os

import sys
import cv2
print('Python %s on %s' % (sys.version, sys.platform))
print('opencv version: %s' % cv2.__version__)


#
# various parameters:
#

prm_gate_size_in_pix = 200
prm_frame_thickness_to_gate_size_ratio = 0.22

prm_frame_thickness_in_pix = prm_gate_size_in_pix * prm_frame_thickness_to_gate_size_ratio
prm_frame_padding_in_pix = prm_gate_size_in_pix * 1./5.
gate_padding_in_pix = prm_frame_padding_in_pix + prm_frame_thickness_in_pix

prm_frame_size_in_pix = prm_gate_size_in_pix + 2 * prm_frame_thickness_in_pix
prm_rectified_gate_img_size_in_pix = int(prm_frame_size_in_pix + 2 * prm_frame_padding_in_pix + 0.5)


prm_ugly_scale_ratio = 0.1
prm_bad_scale_ratio = 0.45
prm_good_scale_ratio = 0.6

prm_ugly_shear = 1.0
prm_bad_shear = 0.6
prm_good_shear = 0.5


prm_rnd_seed = 0  # None

prm_scale_min = 0.4
prm_scale_max = 2.
prm_sxsy_ratio_sigma = 0.002
prm_max_rot_angle_in_rads = 20. * math.pi / 180.
prm_max_normal_coeff = 100.
prm_result_width = 640
prm_result_height = 480

#
# global variables:
#

g_gate_coords = np.array((
    (gate_padding_in_pix, gate_padding_in_pix),
    (gate_padding_in_pix + prm_gate_size_in_pix, gate_padding_in_pix),
    (gate_padding_in_pix + prm_gate_size_in_pix, gate_padding_in_pix + prm_gate_size_in_pix),
    (gate_padding_in_pix, gate_padding_in_pix + prm_gate_size_in_pix),
), dtype="float32")

g_frame_coords = np.array((
    (prm_frame_padding_in_pix, prm_frame_padding_in_pix),
    (prm_frame_padding_in_pix + prm_frame_size_in_pix, prm_frame_padding_in_pix),
    (prm_frame_padding_in_pix + prm_frame_size_in_pix, prm_frame_padding_in_pix + prm_frame_size_in_pix),
    (prm_frame_padding_in_pix, prm_frame_padding_in_pix + prm_frame_size_in_pix),
), dtype="float32")


g_dict_of_labels = {}
g_work_list_of_images = []
g_path_to_images = ''


g_rnd = np.random.RandomState(seed=prm_rnd_seed)


#
# setup and loading functions:
#

def set_rnd_seed(seed):
    global prm_rnd_seed, g_rnd
    prm_rnd_seed = seed
    g_rnd = np.random.RandomState(seed=prm_rnd_seed)


def load_labels(path_to_images, path_to_labels):
    """
    returns preprocessed dict_of_labels and list of image file names
    """
    # : loading list of file names and labels dict:
    with os.scandir(path_to_images) as it:
        set_of_image_file_names = {entry.name for entry in it
                                   if entry.is_file()
                                   and not entry.name.startswith('.')
                                   and entry.name.endswith(('.jpg', '.JPG'))
                                   }

    with open(path_to_labels) as f:
        dict_of_labels = json.load(f)

    set_of_images_with_labels = set_of_image_file_names & dict_of_labels.keys()
    print('found {} images and {} labels.'.format(len(set_of_image_file_names), len(dict_of_labels)))
    print('{} images with labels'.format(len(set_of_images_with_labels)))
    print('{} images without labels'.format(len(set_of_image_file_names - dict_of_labels.keys())))
    print('{} labels without images'.format(len(dict_of_labels.keys() - set_of_image_file_names)))

    dict_of_incomplete_labels = {file_name: label for (file_name, label) in dict_of_labels.items()
                                 if len(label) != 1 or len(label[0]) != 8}
    print('{} incomplete labels:'.format(len(dict_of_incomplete_labels)))
    print(dict_of_incomplete_labels)

    set_of_images_with_labels = set_of_images_with_labels - dict_of_incomplete_labels.keys()
    dict_of_labels = {file_name: label[0] for (file_name, label) in dict_of_labels.items()
                      if file_name in set_of_images_with_labels}
    print('{} images with complete labels'.format(len(set_of_images_with_labels)))
    work_list_of_images = list(set_of_images_with_labels)
    work_list_of_images.sort()
    global g_dict_of_labels, g_work_list_of_images, g_path_to_images
    g_dict_of_labels, g_work_list_of_images, g_path_to_images = dict_of_labels, work_list_of_images, path_to_images
    return dict_of_labels, work_list_of_images


#
# primitive functions to get the stuff:
#

def get_labeled_gate_coords(label):
    """
    takes label as a list of exactly 8 numbers as they are given in herox labels:
    four pairs of (x,y)-coordinates of clockwise ordered corners.
    returns the coordinates as np.array of shape (4,2)
    :param label: four pairs of (x,y)-coordinates of clockwise ordered corner
    :return: coordinates as np.array of shape (4,2)
    """
    return np.array((
        (label[0], label[1]),
        (label[2], label[3]),
        (label[4], label[5]),
        (label[6], label[7]),
    ), dtype="float32")


def label_coords_to_herox_label(label_coords):
    """
    inverse to get_labeled_gate_coords()
    :param label_coords:
    :return:
    """
    return list(np.int32(np.rint(label_coords.ravel())).tolist())


def get_img2gate_homography(labeled_gate_coords):
    return cv2.getPerspectiveTransform(labeled_gate_coords, g_gate_coords)


def get_gate2img_homography(labeled_gate_coords):
    return cv2.getPerspectiveTransform(g_gate_coords, labeled_gate_coords)


def get_transformed_label(label_coords, homography):
    return cv2.perspectiveTransform(np.float32([label_coords]), homography)


def get_gate_frame_label(gate2img_homography):
    return get_transformed_label(g_frame_coords, gate2img_homography)


def get_image_and_gate_coords(image_file_name):
    image = cv2.imread(os.path.join(g_path_to_images, image_file_name))
    image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_bw = cv2.cvtColor(image_bw, cv2.COLOR_GRAY2BGR)
    label = g_dict_of_labels[image_file_name]
    labeled_gate_coords = get_labeled_gate_coords(label)
    return image_bw, labeled_gate_coords


def get_label_center_coords(label_coords):
    # let's call label_center_coords the intersection of diagonals of the label:
    homogeneous_label_coords = cv2.convertPointsToHomogeneous(label_coords)
    line1_h_coords = np.cross(homogeneous_label_coords[0], homogeneous_label_coords[2])
    line2_h_coords = np.cross(homogeneous_label_coords[1], homogeneous_label_coords[3])
    ip_h_coords = np.cross(line1_h_coords, line2_h_coords)
    return cv2.convertPointsFromHomogeneous(ip_h_coords)[0][0]


#
# estimation of camera invariants for rough label quality evaluation:
#

def homography_to_2d_sx_sy_shear(h):
    """
    takes 3x3 homography matrics h
    returns (scale_x, scale_y, shear)
    :param h: 3x3 homography matrics
    :return: sx, sy, shear
    """

    '''
    1. (taken from https://stackoverflow.com/a/34399034/1027013)
        If whole homography is:

        h1 h2 h3
        h4 h5 h6
        h7 h8 1

        then its normalized 2x2 part:

        A = Q*R = [[a,b];[c,d]] =
        [[h1-(h7*h3)   h2-(h8*h3)]
         [h4-(h7*h6)   h5-(h8*h6)]]
    '''
    a = h[0, 0] - h[2, 0] * h[0, 2]
    b = h[0, 1] - h[2, 1] * h[0, 2]
    c = h[1, 0] - h[2, 0] * h[1, 2]
    d = h[1, 1] - h[2, 1] * h[1, 2]

    '''
    2. (taken from https://stackoverflow.com/a/16085365/1027013)
    '''
    det_a = a * d - b * c
    sx = math.sqrt(a * a + b * b)
    sy = det_a / sx
    # fixme: elaborate on shear extraction!
    shear = (a * c + b * d) / det_a
    return sx, sy, shear


def homography_to_2d_cam_invariants(h):
    """
    takes 3x3 homography matrics h
    returns (scale_x/scale_y, abs(shear))
    :param h: 3x3 homography matrics
    :return: sx / sy, normalized shear
    """
    sx, sy, shear = homography_to_2d_sx_sy_shear(h)
    # fixme: elaborate on shear normalization!
    return sx / sy, abs(shear / sy)


def image_file_name_to_2d_cam_invariants(image_file_name):
    label = g_dict_of_labels[image_file_name]
    labled_gate_coords = get_labeled_gate_coords(label)
    h = get_img2gate_homography(labled_gate_coords)
    return homography_to_2d_cam_invariants(h)


# separate images into good, bad, ugly, invalid categories:

def is_invalid_scale_ratio(scale_ratio):
    return scale_ratio < 0  # case of self-intersecting label


def is_ugly_scale_ratio(scale_ratio):
    return (0 < scale_ratio < prm_ugly_scale_ratio or
            scale_ratio > 1 / prm_ugly_scale_ratio)


def is_bad_scale_ratio(scale_ratio):
    return (not is_ugly_scale_ratio(scale_ratio) and
            (scale_ratio < prm_bad_scale_ratio or
             scale_ratio > 1 / prm_bad_scale_ratio))


def is_good_scale_ratio(scale_ratio):
    return prm_good_scale_ratio < scale_ratio < 1 / prm_good_scale_ratio


def is_border_scale_ratio(scale_ratio):
    return (prm_bad_scale_ratio <= scale_ratio <= prm_good_scale_ratio or
            1. / prm_good_scale_ratio <= scale_ratio <= 1. / prm_bad_scale_ratio)


def is_ugly_shear(shear):
    return shear > prm_ugly_shear


def is_bad_shear(shear):
    return prm_bad_shear < shear < prm_ugly_shear


def is_good_shear(shear):
    return shear < prm_good_shear


def is_border_shear(shear):
    return prm_good_shear <= shear <= prm_bad_shear


def describe_scale_ratio(scale_ratio):
    if is_invalid_scale_ratio(scale_ratio):
        return 'invalid'
    elif is_ugly_scale_ratio(scale_ratio):
        return 'ugly'
    elif is_bad_scale_ratio(scale_ratio):
        return 'bad'
    elif is_good_scale_ratio(scale_ratio):
        return 'good'
    else:
        return 'bad?'


def describe_shear(shear):
    if is_ugly_shear(shear):
        return 'ugly'
    elif is_bad_shear(shear):
        return 'bad'
    elif is_good_shear(shear):
        return 'good'
    else:
        return 'bad?'


def describe_scale_ratio_and_shear(scale_ratio, shear):
    return 'cam invariants: sx/sy={}({}), shear={}({})'.format(scale_ratio,
                                                               describe_scale_ratio(scale_ratio),
                                                               shear,
                                                               describe_shear(shear))


def generate_lists_of_image_file_names(work_list_of_images=None,
                                       good=True,
                                       not_good=True,
                                       invalid=True,
                                       gray_zone=True,
                                       ugly=True,
                                       bad=True):
    result = {}
    if not work_list_of_images:
        work_list_of_images = g_work_list_of_images
    cam_invariants = [image_file_name_to_2d_cam_invariants(image_file_name)
                      for image_file_name in work_list_of_images]

    if good:
        result['good'] = [image_file_name for i, image_file_name in enumerate(work_list_of_images)
                          if is_good_scale_ratio(cam_invariants[i][0]) and is_good_shear(cam_invariants[i][1])]
    if not_good:
        result['not_good'] = [image_file_name for i, image_file_name in enumerate(work_list_of_images)
                              if not is_good_scale_ratio(cam_invariants[i][0])
                              or not is_good_shear(cam_invariants[i][1])]
    if invalid:
        result['invalid'] = [image_file_name for i, image_file_name in enumerate(work_list_of_images)
                             if is_invalid_scale_ratio(cam_invariants[i][0])]
    if gray_zone:
        result['gray_zone'] = [image_file_name for i, image_file_name in enumerate(work_list_of_images)
                               if is_border_scale_ratio(cam_invariants[i][0])
                               or is_border_shear(cam_invariants[i][1])]
    if ugly:
        result['ugly'] = [image_file_name for i, image_file_name in enumerate(work_list_of_images)
                          if is_ugly_scale_ratio(cam_invariants[i][0])
                          or is_ugly_shear(cam_invariants[i][1])]
    if bad:
        result['bad'] = [image_file_name for i, image_file_name in enumerate(work_list_of_images)
                         if is_bad_scale_ratio(cam_invariants[i][0])
                         or is_bad_shear(cam_invariants[i][1])]
    return result


#
# primitive functions to draw the stuff:
#


def draw_label(image, label_coords, color=(255, 0, 255), thickness=3):
    cv2.polylines(image,
                  np.int32([label_coords]),
                  isClosed=True, color=color, thickness=thickness, lineType=8)
    return image


def draw_gate_labels(image, labled_gate_coords):
    # draw the gate label:
    image = draw_label(image, labled_gate_coords, color=(255, 0, 255), thickness=3)
    # draw the gate's frame expected corners:
    gate2img_homography = get_gate2img_homography(labled_gate_coords)
    frame_label_coords = get_gate_frame_label(gate2img_homography)
    image = draw_label(image, frame_label_coords, color=(0, 255, 0), thickness=2)
    return image


def get_rectified_gate_image(image, img2gate_homography):
    return cv2.warpPerspective(image,
                               img2gate_homography,
                               (prm_rectified_gate_img_size_in_pix,
                                prm_rectified_gate_img_size_in_pix),
                               borderValue=(0, 0, 255))


def make_exploration_images(image_label, labeled_gate_coords):
    image_label = draw_gate_labels(image_label, labeled_gate_coords)
    # make 'rectified' image of the gate:
    img2gate_homography = get_img2gate_homography(labeled_gate_coords)
    image_label_rectified = get_rectified_gate_image(image_label, img2gate_homography)
    return image_label, image_label_rectified


def make_stacked_exploration_image(image_label, labled_gate_coords):
    image_label, image_label_rectified = make_exploration_images(image_label, labled_gate_coords)
    img2gate_homography = get_img2gate_homography(labled_gate_coords)
    cam_invs = homography_to_2d_cam_invariants(img2gate_homography)
    cam_invs_as_text = describe_scale_ratio_and_shear(cam_invs[0], cam_invs[1])
    # : stack them horizontally:
    text_height = 30
    res_h = image_label_rectified.shape[0] + text_height
    label_image_scale = float(image_label_rectified.shape[0]) / image_label.shape[0]
    image_label = cv2.resize(image_label, None, fx=label_image_scale, fy=label_image_scale)
    div_w = 2
    label_image_w = image_label.shape[1]
    res_w = label_image_w + div_w + image_label_rectified.shape[1]
    stacked_images = np.zeros((res_h, res_w, 3), dtype="uint8")
    stacked_images[:-text_height, :label_image_w, :] = image_label
    stacked_images[:-text_height, label_image_w + div_w:, :] = image_label_rectified
    cv2.putText(stacked_images,
                cam_invs_as_text,
                (10, res_h - 6),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.4,
                thickness=1,
                color=(200, 255, 155),
                bottomLeftOrigin=False)
    return stacked_images


#
# functions for writing the stuff:
#

def write_exploration_images(list_of_image_file_names, path_to_results):
    os.makedirs(path_to_results, exist_ok=True)
    for image_file_name in list_of_image_file_names:
        result_image, labled_gate_coords = get_image_and_gate_coords(image_file_name)
        result_image = make_stacked_exploration_image(result_image, labled_gate_coords)
        cv2.imwrite(os.path.join(path_to_results, image_file_name), result_image)


def get_frame_label(image_file_name):
    label = g_dict_of_labels[image_file_name]
    labled_gate_coords = get_labeled_gate_coords(label)
    gate2img_homography = get_gate2img_homography(labled_gate_coords)
    frame_label_coords = get_gate_frame_label(gate2img_homography)
    return label_coords_to_herox_label(frame_label_coords)


def generate_frame_labels(list_of_image_file_names, output_to_json_file_path=None):
    result = {
        image_file_name: [get_frame_label(image_file_name)]
        for image_file_name in list_of_image_file_names
    }
    if output_to_json_file_path:
        with open(output_to_json_file_path, "w") as write_file:
            json.dump(result, write_file)
    return result


#
# functions for augmentation of the dataset
#

def make_homography(sx, sy, shear, rot, nx, ny):
    ns = math.sqrt(nx*nx + ny*ny + 1)
    pr = np.array((
        (ns, 0,  0),
        (0,  ns,  0),
        (nx, ny, 1),
    ), dtype="float32")
    sc = np.array((
        (sx, 0,  0),
        (0,  sy, 0),
        (0,  0,  1),
    ), dtype="float32")
    sh = np.array((
        (1, math.sin(shear), 0),
        (0, math.cos(shear), 0),
        (0, 0, 1),
    ), dtype="float32")
    sin_r = math.sin(rot)
    cos_r = math.cos(rot)
    rt = np.array((
        (cos_r, -sin_r, 0),
        (sin_r,  cos_r, 0),
        (0, 0, 1),
    ), dtype="float32")
    return pr @ sc @ sh @ rt


def offset_homography(homography, mx, my):
    mv = np.array((
        (1, 0, mx),
        (0, 1, my),
        (0, 0, 1),
    ), dtype="float32")
    return mv @ homography


def make_centered_homography(label_coords, sx, sy, shear, rot, nx, ny):
    homography = make_homography(sx, sy, shear, rot, nx, ny)
    reprojected_label_coords = cv2.perspectiveTransform(np.float32([label_coords]), homography)
    reprojected_label_center_coords = get_label_center_coords(reprojected_label_coords)
    mx = prm_result_width / 2. - reprojected_label_center_coords[0]
    my = prm_result_height / 2. - reprojected_label_center_coords[1]
    return offset_homography(homography, mx, my)


def get_random_sxsy():
    # assume log(scale) is uniformly distributed over [log(prm_scale_min), log(prm_scale_max)]
    scale = math.exp(g_rnd.uniform(low=math.log(prm_scale_min), high=math.log(prm_scale_max)))
    sxsy_ratio = g_rnd.normal(scale=prm_sxsy_ratio_sigma)
    # sqrt(sx*sy)=scale, sx/sy=sxsy_ratio
    # => sy=scale/sqrt(sxsy_ratio), sx=sxsy_ratio*sy=scale*sqrt(sxsy_ratio)
    sx = scale * math.sqrt(sxsy_ratio)
    sy = scale / math.sqrt(sxsy_ratio)
    return sx, sy


def get_offset_ranges(points, homography):
    reprojected_points = cv2.perspectiveTransform(np.float32([points]), homography)
    # find the xy-ranges of translations which send the reprojected_points
    #   into the image of size (prm_result_width, prm_result_height):
    return reprojected_points


def make_warped_centered_image_and_label(image_file_name, sx=1., sy=1., shear=0., rot=0., nx=0., ny=0.):
    result_image, gate_coords = get_image_and_gate_coords(image_file_name)
    warping_h = make_centered_homography(gate_coords, sx, sy, shear, rot, nx, ny)
    reprojected_gate_coords = cv2.perspectiveTransform(np.float32([gate_coords]), warping_h)
    result_image = cv2.warpPerspective(result_image,
                                       warping_h,
                                       (prm_result_width,
                                        prm_result_height),
                                       borderValue=(0, 0, 255))
    result_image = make_stacked_exploration_image(result_image, reprojected_gate_coords)
    return result_image, reprojected_gate_coords

