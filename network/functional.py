import math
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from utils.config import cfg

""" Utilities for computing, reading and saving benchmark evaluation."""


def Norm2d(in_channels):
    """
    Custom Norm Function to allow flexible switching
    """
    layer = getattr(cfg.MODEL, 'BNFUNC')
    normalizationLayer = layer(in_channels)
    return normalizationLayer


def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


def eval_mask_boundary(seg_mask, gt_mask, num_classes, num_proc=10, bound_th=0.008):
    """
    Compute F score for a segmentation mask

    Arguments:
        seg_mask (ndarray): segmentation mask prediction
        gt_mask (ndarray): segmentation mask ground truth
        num_classes (int): number of classes

    Returns:
        F (float): mean F score across all classes
        Fpc (listof float): F score per class
    """
    p = Pool(processes=num_proc)
    batch_size = seg_mask.shape[0]

    Fpc = np.zeros(num_classes)
    Fc = np.zeros(num_classes)
    for class_id in tqdm(range(num_classes)):
        args = [((seg_mask[i] == class_id).astype(np.uint8),
                 (gt_mask[i] == class_id).astype(np.uint8),
                 gt_mask[i] == 255,
                 bound_th)
                for i in range(batch_size)]
        temp = p.map(db_eval_boundary_wrapper, args)
        temp = np.array(temp)
        Fs = temp[:, 0]
        _valid = ~np.isnan(Fs)
        Fc[class_id] = np.sum(_valid)
        Fs[np.isnan(Fs)] = 0
        Fpc[class_id] = sum(Fs)
    return Fpc, Fc


# def db_eval_boundary_wrapper_wrapper(args):
#    seg_mask, gt_mask, class_id, batch_size, Fpc = args
#    print("class_id:" + str(class_id))
#    p = Pool(processes=10)
#    args = [((seg_mask[i] == class_id).astype(np.uint8),
#             (gt_mask[i] == class_id).astype(np.uint8))
#             for i in range(batch_size)]
#    Fs = p.map(db_eval_boundary_wrapper, args)
#    Fpc[class_id] = sum(Fs)
#    return

def db_eval_boundary_wrapper(args):
    foreground_mask, gt_mask, ignore, bound_th = args
    return db_eval_boundary(foreground_mask, gt_mask, ignore, bound_th)


def db_eval_boundary(foreground_mask, gt_mask, ignore_mask, bound_th=0.008):
    """
    Compute mean,recall and decay from per-frame evaluation.
    Calculates precision/recall for boundaries between foreground_mask and
    gt_mask using morphological operators to speed it up.

    Arguments:
        foreground_mask (ndarray): binary segmentation image.
        gt_mask         (ndarray): binary annotated image.

    Returns:
        F (float): boundaries F-measure
        P (float): boundaries precision
        R (float): boundaries recall
    """
    assert np.atleast_3d(foreground_mask).shape[2] == 1

    bound_pix = bound_th if bound_th >= 1 else \
        np.ceil(bound_th * np.linalg.norm(foreground_mask.shape))

    # print(bound_pix)
    # print(gt.shape)
    # print(np.unique(gt))
    foreground_mask[ignore_mask] = 0
    gt_mask[ignore_mask] = 0

    # Get the pixel boundaries of both masks
    fg_boundary = seg2bmap(foreground_mask);
    gt_boundary = seg2bmap(gt_mask);

    from skimage.morphology import binary_dilation, disk

    fg_dil = binary_dilation(fg_boundary, disk(bound_pix))
    gt_dil = binary_dilation(gt_boundary, disk(bound_pix))

    # Get the intersection
    gt_match = gt_boundary * fg_dil
    fg_match = fg_boundary * gt_dil

    # Area of the intersection
    n_fg = np.sum(fg_boundary)
    n_gt = np.sum(gt_boundary)

    # % Compute precision and recall
    if n_fg == 0 and n_gt > 0:
        precision = 1
        recall = 0
    elif n_fg > 0 and n_gt == 0:
        precision = 0
        recall = 1
    elif n_fg == 0 and n_gt == 0:
        precision = 1
        recall = 1
    else:
        precision = np.sum(fg_match) / float(n_fg)
        recall = np.sum(gt_match) / float(n_gt)

    # Compute F measure
    if precision + recall == 0:
        F = 0
    else:
        F = 2 * precision * recall / (precision + recall);

    return F, precision


def seg2bmap(seg, width=None, height=None):
    """
    From a segmentation, compute a binary boundary map with 1 pixel wide
    boundaries.  The boundary pixels are offset by 1/2 pixel towards the
    origin from the actual segment boundary.

    Arguments:
        seg     : Segments labeled from 1..k.
        width	  :	Width of desired bmap  <= seg.shape[1]
        height  :	Height of desired bmap <= seg.shape[0]

    Returns:
        bmap (ndarray):	Binary boundary map.

     David Martin <dmartin@eecs.berkeley.edu>
     January 2003
 """

    seg = seg.astype(np.bool)
    seg[seg > 0] = 1

    assert np.atleast_3d(seg).shape[2] == 1

    width = seg.shape[1] if width is None else width
    height = seg.shape[0] if height is None else height

    h, w = seg.shape[:2]

    ar1 = float(width) / float(height)
    ar2 = float(w) / float(h)

    assert not (width > w | height > h | abs(ar1 - ar2) > 0.01), \
        'Can''t convert %dx%d seg to %dx%d bmap.' % (w, h, width, height)

    e = np.zeros_like(seg)
    s = np.zeros_like(seg)
    se = np.zeros_like(seg)

    e[:, :-1] = seg[:, 1:]
    s[:-1, :] = seg[1:, :]
    se[:-1, :-1] = seg[1:, 1:]

    b = seg ^ e | seg ^ s | seg ^ se
    b[-1, :] = seg[-1, :] ^ e[-1, :]
    b[:, -1] = seg[:, -1] ^ s[:, -1]
    b[-1, -1] = 0

    if w == width and h == height:
        bmap = b
    else:
        bmap = np.zeros((height, width))
        for x in range(w):
            for y in range(h):
                if b[y, x]:
                    j = 1 + math.floor((y - 1) + height / h)
                    i = 1 + math.floor((x - 1) + width / h)
                    bmap[j, i] = 1

    return bmap


def calc_pad_same(in_siz, out_siz, stride, ksize):
    """Calculate same padding width.
    Args:
    ksize: kernel size [I, J].
    Returns:
    pad_: Actual padding width.
    """
    return (out_siz - 1) * stride + ksize - in_siz


def conv2d_same(input, kernel, groups,bias=None,stride=1,padding=0,dilation=1):
    n, c, h, w = input.shape
    kout, ki_c_g, kh, kw = kernel.shape
    pw = calc_pad_same(w, w, 1, kw)
    ph = calc_pad_same(h, h, 1, kh)
    pw_l = pw // 2
    pw_r = pw - pw_l
    ph_t = ph // 2
    ph_b = ph - ph_t

    input_ = F.pad(input, (pw_l, pw_r, ph_t, ph_b))
    result = F.conv2d(input_, kernel, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
    assert result.shape == input.shape
    return result


def gradient_central_diff(input, cuda):
    return input, input
    kernel = [[1, 0, -1]]
    kernel_t = 0.5 * torch.Tensor(kernel) * -1.  # pytorch implements correlation instead of conv
    if type(cuda) is int:
        if cuda != -1:
            kernel_t = kernel_t.cuda(device=cuda)
    else:
        if cuda is True:
            kernel_t = kernel_t.cuda()
    n, c, h, w = input.shape

    x = conv2d_same(input, kernel_t.unsqueeze(0).unsqueeze(0).repeat([c, 1, 1, 1]), c)
    y = conv2d_same(input, kernel_t.t().unsqueeze(0).unsqueeze(0).repeat([c, 1, 1, 1]), c)
    return x, y


def compute_single_sided_diferences(o_x, o_y, input):
    # n,c,h,w
    #input = input.clone()
    o_y[:, :, 0, :] = input[:, :, 1, :].clone() - input[:, :, 0, :].clone()
    o_x[:, :, :, 0] = input[:, :, :, 1].clone() - input[:, :, :, 0].clone()
    # --
    o_y[:, :, -1, :] = input[:, :, -1, :].clone() - input[:, :, -2, :].clone()
    o_x[:, :, :, -1] = input[:, :, :, -1].clone() - input[:, :, :, -2].clone()
    return o_x, o_y


def numerical_gradients_2d(input, cuda=False):
    """
    numerical gradients implementation over batches using torch group conv operator.
    the single sided differences are re-computed later.
    it matches np.gradient(image) with the difference than here output=x,y for an image while there output=y,x
    :param input: N,C,H,W
    :param cuda: whether or not use cuda
    :return: X,Y
    """
    n, c, h, w = input.shape
    assert h > 1 and w > 1
    x, y = gradient_central_diff(input, cuda)
    return x, y


def convTri(input, r, cuda=False):
    """
    Convolves an image by a 2D triangle filter (the 1D triangle filter f is
    [1:r r+1 r:-1:1]/(r+1)^2, the 2D version is simply conv2(f,f'))
    :param input:
    :param r: integer filter radius
    :param cuda: move the kernel to gpu
    :return:
    """
    if (r <= 1):
        raise ValueError()
    n, c, h, w = input.shape
    return input
    f = list(range(1, r + 1)) + [r + 1] + list(reversed(range(1, r + 1)))
    kernel = torch.Tensor([f]) / (r + 1) ** 2
    if type(cuda) is int:
        if cuda != -1:
            kernel = kernel.cuda(device=cuda)
    else:
        if cuda is True:
            kernel = kernel.cuda()

    # padding w
    input_ = F.pad(input, (1, 1, 0, 0), mode='replicate')
    input_ = F.pad(input_, (r, r, 0, 0), mode='reflect')
    input_ = [input_[:, :, :, :r], input, input_[:, :, :, -r:]]
    input_ = torch.cat(input_, 3)
    t = input_

    # padding h
    input_ = F.pad(input_, (0, 0, 1, 1), mode='replicate')
    input_ = F.pad(input_, (0, 0, r, r), mode='reflect')
    input_ = [input_[:, :, :r, :], t, input_[:, :, -r:, :]]
    input_ = torch.cat(input_, 2)

    output = F.conv2d(input_,
                      kernel.unsqueeze(0).unsqueeze(0).repeat([c, 1, 1, 1]),
                      padding=0, groups=c)
    output = F.conv2d(output,
                      kernel.t().unsqueeze(0).unsqueeze(0).repeat([c, 1, 1, 1]),
                      padding=0, groups=c)
    return output


def compute_normal(E, cuda=False):
    if torch.sum(torch.isnan(E)) != 0:
        print('nans found here')
        import ipdb;
        ipdb.set_trace()
    E_ = convTri(E, 4, cuda)
    Ox, Oy = numerical_gradients_2d(E_, cuda)
    Oxx, _ = numerical_gradients_2d(Ox, cuda)
    Oxy, Oyy = numerical_gradients_2d(Oy, cuda)

    aa = Oyy * torch.sign(-(Oxy + 1e-5)) / (Oxx + 1e-5)
    t = torch.atan(aa)
    O = torch.remainder(t, np.pi)

    if torch.sum(torch.isnan(O)) != 0:
        print('nans found here')
        import ipdb;
        ipdb.set_trace()

    return O


def compute_normal_2(E, cuda=False):
    if torch.sum(torch.isnan(E)) != 0:
        print('nans found here')
        import ipdb;
        ipdb.set_trace()
    E_ = convTri(E, 4, cuda)
    Ox, Oy = numerical_gradients_2d(E_, cuda)
    Oxx, _ = numerical_gradients_2d(Ox, cuda)
    Oxy, Oyy = numerical_gradients_2d(Oy, cuda)

    aa = Oyy * torch.sign(-(Oxy + 1e-5)) / (Oxx + 1e-5)
    t = torch.atan(aa)
    O = torch.remainder(t, np.pi)

    if torch.sum(torch.isnan(O)) != 0:
        print('nans found here')
        import ipdb;
        ipdb.set_trace()

    return O, (Oyy, Oxx)


def compute_grad_mag(E, cuda=False):
    E_ = convTri(E, 4, cuda)
    Ox, Oy = numerical_gradients_2d(E_, cuda)
    mag = torch.sqrt(torch.mul(Ox,Ox) + torch.mul(Oy,Oy) + 1e-6)
    mag = mag / mag.max();

    return mag


def eval_mask_boundary(seg_mask, gt_mask, num_classes, num_proc=10, bound_th=0.008):
    """
    Compute F score for a segmentation mask

    Arguments:
        seg_mask (ndarray): segmentation mask prediction
        gt_mask (ndarray): segmentation mask ground truth
        num_classes (int): number of classes

    Returns:
        F (float): mean F score across all classes
        Fpc (listof float): F score per class
    """
    p = Pool(processes=num_proc)
    batch_size = seg_mask.shape[0]

    Fpc = np.zeros(num_classes)
    Fc = np.zeros(num_classes)
    for class_id in tqdm(range(num_classes)):
        args = [((seg_mask[i] == class_id).astype(np.uint8),
                 (gt_mask[i] == class_id).astype(np.uint8),
                 gt_mask[i] == 255,
                 bound_th)
                for i in range(batch_size)]
        temp = p.map(db_eval_boundary_wrapper, args)
        temp = np.array(temp)
        Fs = temp[:, 0]
        _valid = ~np.isnan(Fs)
        Fc[class_id] = np.sum(_valid)
        Fs[np.isnan(Fs)] = 0
        Fpc[class_id] = sum(Fs)
    return Fpc, Fc


# def db_eval_boundary_wrapper_wrapper(args):
#    seg_mask, gt_mask, class_id, batch_size, Fpc = args
#    print("class_id:" + str(class_id))
#    p = Pool(processes=10)
#    args = [((seg_mask[i] == class_id).astype(np.uint8),
#             (gt_mask[i] == class_id).astype(np.uint8))
#             for i in range(batch_size)]
#    Fs = p.map(db_eval_boundary_wrapper, args)
#    Fpc[class_id] = sum(Fs)
#    return

def db_eval_boundary_wrapper(args):
    foreground_mask, gt_mask, ignore, bound_th = args
    return db_eval_boundary(foreground_mask, gt_mask, ignore, bound_th)


def db_eval_boundary(foreground_mask, gt_mask, ignore_mask, bound_th=0.008):
    """
    Compute mean,recall and decay from per-frame evaluation.
    Calculates precision/recall for boundaries between foreground_mask and
    gt_mask using morphological operators to speed it up.

    Arguments:
        foreground_mask (ndarray): binary segmentation image.
        gt_mask         (ndarray): binary annotated image.

    Returns:
        F (float): boundaries F-measure
        P (float): boundaries precision
        R (float): boundaries recall
    """
    assert np.atleast_3d(foreground_mask).shape[2] == 1

    bound_pix = bound_th if bound_th >= 1 else \
        np.ceil(bound_th * np.linalg.norm(foreground_mask.shape))

    # print(bound_pix)
    # print(gt.shape)
    # print(np.unique(gt))
    foreground_mask[ignore_mask] = 0
    gt_mask[ignore_mask] = 0

    # Get the pixel boundaries of both masks
    fg_boundary = seg2bmap(foreground_mask);
    gt_boundary = seg2bmap(gt_mask);

    from skimage.morphology import binary_dilation, disk

    fg_dil = binary_dilation(fg_boundary, disk(bound_pix))
    gt_dil = binary_dilation(gt_boundary, disk(bound_pix))

    # Get the intersection
    gt_match = gt_boundary * fg_dil
    fg_match = fg_boundary * gt_dil

    # Area of the intersection
    n_fg = np.sum(fg_boundary)
    n_gt = np.sum(gt_boundary)

    # % Compute precision and recall
    if n_fg == 0 and n_gt > 0:
        precision = 1
        recall = 0
    elif n_fg > 0 and n_gt == 0:
        precision = 0
        recall = 1
    elif n_fg == 0 and n_gt == 0:
        precision = 1
        recall = 1
    else:
        precision = np.sum(fg_match) / float(n_fg)
        recall = np.sum(gt_match) / float(n_gt)

    # Compute F measure
    if precision + recall == 0:
        F = 0
    else:
        F = 2 * precision * recall / (precision + recall);

    return F, precision


def seg2bmap(seg, width=None, height=None):
    """
    From a segmentation, compute a binary boundary map with 1 pixel wide
    boundaries.  The boundary pixels are offset by 1/2 pixel towards the
    origin from the actual segment boundary.

    Arguments:
        seg     : Segments labeled from 1..k.
        width	  :	Width of desired bmap  <= seg.shape[1]
        height  :	Height of desired bmap <= seg.shape[0]

    Returns:
        bmap (ndarray):	Binary boundary map.

     David Martin <dmartin@eecs.berkeley.edu>
     January 2003
 """

    seg = seg.astype(np.bool)
    seg[seg > 0] = 1

    assert np.atleast_3d(seg).shape[2] == 1

    width = seg.shape[1] if width is None else width
    height = seg.shape[0] if height is None else height

    h, w = seg.shape[:2]

    ar1 = float(width) / float(height)
    ar2 = float(w) / float(h)

    assert not (width > w | height > h | abs(ar1 - ar2) > 0.01), \
        'Can''t convert %dx%d seg to %dx%d bmap.' % (w, h, width, height)

    e = np.zeros_like(seg)
    s = np.zeros_like(seg)
    se = np.zeros_like(seg)

    e[:, :-1] = seg[:, 1:]
    s[:-1, :] = seg[1:, :]
    se[:-1, :-1] = seg[1:, 1:]

    b = seg ^ e | seg ^ s | seg ^ se
    b[-1, :] = seg[-1, :] ^ e[-1, :]
    b[:, -1] = seg[:, -1] ^ s[:, -1]
    b[-1, -1] = 0

    if w == width and h == height:
        bmap = b
    else:
        bmap = np.zeros((height, width))
        for x in range(w):
            for y in range(h):
                if b[y, x]:
                    j = 1 + math.floor((y - 1) + height / h)
                    i = 1 + math.floor((x - 1) + width / h)
                    bmap[j, i] = 1

    return bmap
