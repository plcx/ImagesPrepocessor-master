

import os
import numpy as np
import matplotlib.pyplot as plt
from monai.transforms.utils import morphology, ndimage
from scipy import ndimage as ndi
from skimage.morphology import binary_erosion

from utils.data_io import nib_load, nib_save
from skimage.feature import peak_local_max
from skimage.segmentation import watershed


import glob
from scipy import ndimage, stats
from skimage.morphology import h_maxima, binary_opening
from skimage.segmentation import watershed
from scipy.ndimage import binary_closing
import nibabel as nib
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
import time
import pandas as pd




def set_boundary_zero(pre_seg):
    '''
    SET_BOUNARY_ZERO is used to set all segmented regions attached to the boundary as zero background.
    :param pre_seg:
    :return:
    '''
    opened_mask = binary_opening(pre_seg)
    pre_seg[opened_mask == 0] = 0
    seg_shape = pre_seg.shape
    boundary_mask = np.zeros_like(pre_seg, dtype=np.uint8)
    boundary_mask[0:2, :, :] = 1;
    boundary_mask[:, 0:2, :] = 1;
    boundary_mask[:, :, 0:2] = 1
    boundary_mask[seg_shape[0] - 1:, :, :] = 1;
    boundary_mask[:, seg_shape[1] - 1:, :] = 1;
    boundary_mask[:, :, seg_shape[2] - 1:] = 1
    boundary_labels = np.unique(pre_seg[boundary_mask != 0])
    for boundary_label in boundary_labels:
        pre_seg[pre_seg == boundary_label] = 0

    return pre_seg

def instance_segmentation_with_nucleus(file, target_files_path, size):
    embryo_name_tp = "_".join(os.path.basename(file).split("_")[0:2])
    segNuc_file = os.path.join(os.getcwd(), 'OutputData', "SegNuc", embryo_name_tp + "_watershellNuc.nii.gz")

    marker = nib.load(segNuc_file).get_fdata()
    print(len(np.unique(marker)))
    marker[marker > 0] = 1

    memb = nib.load(file).get_fdata()

    if (len(np.unique(memb)) == 2):
        image = memb
    else:
        image = np.zeros_like(memb)
        image[memb > 200] = 1
    image = (image == 0).astype(np.uint16)

    struc = np.ones((size, size, size), dtype=bool)
    marker = ndimage.binary_dilation(marker, structure=struc)
    marker_final = ndimage.label(marker)[0]
    print(ndimage.label(marker)[1])

    memb_distance = ndimage.distance_transform_edt(image)
    memb_distance_reverse = memb_distance.max() - memb_distance

    watershed_seg = watershed(memb_distance_reverse, marker_final.astype(np.uint16), watershed_line=True)
    watershed_seg = set_boundary_zero(watershed_seg)

    save_path = os.path.join(target_files_path, embryo_name_tp + "_segcell.nii.gz")
    nib.save(nib.Nifti1Image(watershed_seg.astype(np.uint16), np.eye(4)), save_path)
    print('Finished: ', embryo_name_tp, 'segMemb ---> segCell !')


def instance_segmentation_without_nucleus(file, target_files_path, size):
    embryo_name_tp = "_".join(os.path.basename(file).split("_")[0:2])

    memb = nib.load(file).get_fdata()

    if (len(np.unique(memb)) == 2):
        image = memb
    else:
        image = np.zeros_like(memb)
        image[memb > 200] = 1
    image = (image == 0).astype(np.uint16)

    image1 = ndimage.binary_opening(image).astype(float)
    image2 = ndimage.distance_transform_edt(image1)
    marker = h_maxima(image2, 1)
    print(len(marker[np.nonzero(marker)]))

    struc = np.ones((size, size, size), dtype=bool)
    marker = ndimage.binary_dilation(marker, structure=struc)
    marker_final = ndimage.label(marker)[0]
    print(ndimage.label(marker)[1])

    memb_distance = ndimage.distance_transform_edt(image)
    memb_distance_reverse = memb_distance.max() - memb_distance

    watershed_seg = watershed(memb_distance_reverse, marker_final.astype(np.uint16), watershed_line=True)
    watershed_seg = set_boundary_zero(watershed_seg)

    save_path = os.path.join(target_files_path, embryo_name_tp + "_segcell.nii.gz")
    nib.save(nib.Nifti1Image(watershed_seg.astype(np.uint16), np.eye(4)), save_path)
    print('Finished: ', embryo_name_tp, 'segMemb ---> segCell !')


def remap_labels_for_seg(pred_files_path, gt_files_path):
    pred_files = glob.glob(os.path.join(pred_files_path, "*segcell.nii.gz"))
    gt_files = glob.glob(os.path.join(gt_files_path, "*.nii.gz"))
    assert len(pred_files) == len(gt_files)

    for pred_file in tqdm(pred_files, total=len(pred_files)):
        embryo_name_tp = "_".join(os.path.basename(pred_file).split("_")[0:2])
        gt_file = os.path.join(gt_files_path, embryo_name_tp + "_segCell.nii.gz")
        this_mapping_dict = {}
        pred_embryo = nib.load(pred_file).get_fdata().astype(np.uint16)
        target_embryo = nib.load(gt_file).get_fdata().astype(np.uint16)
        pred2target = pair_labels(pred_embryo, target_embryo)

        target_max = target_embryo.max()
        pred_id_list = list(np.unique(pred_embryo))[1:]
        target_id_list = list(np.unique(target_embryo))[1:]

        out = np.zeros_like(pred_embryo)
        left_labels = pred_id_list.copy()
        for pred_id, target_id in pred2target.items():
            overlap_mask = np.logical_and(pred_embryo == pred_id, target_embryo == target_id)
            if overlap_mask.sum() == 0:
                continue
            left_labels.remove(pred_id)
            out[pred_embryo == pred_id] = target_id
            this_mapping_dict[int(target_id)] = int(pred_id)
        if len(left_labels) > 0:
            for left_label in left_labels:
                target_max += 1
                out[pred_embryo == left_label] = target_max
                this_mapping_dict[int(target_max)] = int(left_label)

        save_path = pred_file.replace(".nii.gz", "_uni.nii.gz")
        nib.save(nib.Nifti1Image(out.astype(np.uint16), np.eye(4)), save_path)
        print('Finished: ', embryo_name_tp, 'segCell ---> Uni_segCell !')




def pair_labels(pred, target):
    """Pairwise the labels between pred and target"""
    target = np.copy(target)  # ? do we need this
    pred = np.copy(pred)
    target_id_list = list(np.unique(target))
    pred_id_list = list(np.unique(pred))
    # print(np.unique(target,return_counts=True))
    # print(np.unique(pred,return_counts=True))

    target_masks = {}
    for t in target_id_list:
        if t==0:
            continue
        t_mask = np.array(target == t, np.uint8)
        a = t_mask.sum()
        target_masks[t]=t_mask

    pred_masks = {}
    pred_dict_id_to_list_order={}
    for tmp_idx_p,p in enumerate(pred_id_list):
        if p==0:
            continue
        p_mask = np.array(pred == p, np.uint8)
        pred_masks[p]=p_mask
        pred_dict_id_to_list_order[p]=tmp_idx_p


    # prefill with value
    pairwise_inter = np.zeros([len(target_id_list) - 1,
                               len(pred_id_list) - 1], dtype=np.float64)
    pairwise_union = np.zeros([len(target_id_list) - 1,
                               len(pred_id_list) - 1], dtype=np.float64)


    # caching pairwise
    for t_idx,target_id in enumerate(target_id_list[1:]):  # 0-th is background
        # print(t_idx,target_id)
        t_mask = target_masks[target_id]
        pred_target_overlap = pred[t_mask > 0]
        pred_target_overlap_id = np.unique(pred_target_overlap)
        pred_target_overlap_id = list(pred_target_overlap_id)

        for pred_id in pred_target_overlap_id:
            if pred_id == 0:  # ignore
                continue
            p_mask = pred_masks[pred_id]
            total = (t_mask + p_mask).sum()
            # overlaping background
            inter = (t_mask * p_mask).sum()
            p_idx=pred_dict_id_to_list_order[pred_id]-1 # t_idx has been -1 for target_id_list[1:]
            pairwise_inter[t_idx, p_idx] = inter
            pairwise_union[t_idx, p_idx] = total - inter
    #
    pairwise_iou = pairwise_inter / (pairwise_union + 1.0e-6)
    # Munkres pairing to find maximal unique pairing
    paired_target, paired_pred = linear_sum_assignment(-pairwise_iou)
    # print(pairwise_iou)
    # print(paired_target)
    # print(paired_pred)
    pred_labels = pred_id_list[1:]
    target_labels = target_id_list[1:]
    pred2target_dict = {pred_labels[pred_idx]:target_labels[target_label_idx] for pred_idx, target_label_idx in zip(paired_pred, paired_target)}

    return pred2target_dict


def iou_dicescore_evaluate(pred_files_path, gt_files_path):
    print("doing evaluation!")
    gt_files = sorted(glob.glob(os.path.join(gt_files_path, "*.nii.gz")))
    pred_files = sorted(glob.glob(os.path.join(pred_files_path, "*_segCell_uni.nii.gz")))
    assert len(pred_files) == len(gt_files), "#gt_files != #pred_files"

    embryo_using_pd_names = []
    all_ious = []
    all_dices = []
    all_cell_label = []
    all_cell_sizes = []

    embryo_ious = []  # average iou
    embryo_dices = []

    for gt_file, pred_file in tqdm(zip(gt_files, pred_files), total=len(gt_files)):
        gt = nib.load(gt_file).get_fdata().astype(np.uint16)
        pred = nib.load(pred_file).get_fdata().astype(np.uint16)

        cells_label, cells_sizes, ious, dices = get_size_dice_and_iou(gt, pred)
        embryo_using_pd_names += [os.path.basename(pred_file).split(".")[0]] * len(cells_sizes)
        all_cell_label += cells_label
        all_cell_sizes += cells_sizes
        all_dices += dices
        all_ious += ious

        embryo_ious.append(sum(ious) / len(ious))
        embryo_dices.append(sum(dices) / len(dices))

    pd_each_cell_score = pd.DataFrame(
        data={"EmbryoName": embryo_using_pd_names, 'CellLabel': all_cell_label, "CellSize": all_cell_sizes,
              'IoU': all_ious, "DiceScore": all_dices})
    # save_file = os.path.join(args["save_folder"], base_data + "_" + os.path.basename(os.path.basename(gt_folder)) + "_score.csv")

    embryo_names = [os.path.basename(basename_path).split('.')[0] for basename_path in pred_files]
    pd_embryo_avg_score = pd.DataFrame(data={'EmbryoName': embryo_names, 'IoU': embryo_ious, 'DiceScore': embryo_dices})

    return pd_each_cell_score, pd_embryo_avg_score

def get_size_dice_and_iou(true, pred):

    true = np.copy(true)
    pred = np.copy(pred)
    # 0:0 1:451 2:536 3:123 4:376 5:878 6:1211 7:4 8:1010 9:1043 10:251 11:1126 12:658 13:755  1027?
    true_ids = list(np.unique(true))
    pred_ids = list(np.unique(pred))
    # remove background aka id 0
    true_ids.remove(0)
    pred_ids.remove(0)
    dices = []
    ious=[]
    target_cell_label=[]
    target_cell_pixels = []
    for true_id in true_ids:
        overlap = np.logical_and(true==true_id, pred==true_id).astype(np.uint8)
        unionlap=np.logical_or(true==true_id, pred==true_id).astype(np.uint8)
        dice = 2 * overlap.sum() / ((true==true_id).sum() + (pred==true_id).sum())
        iou=overlap.sum() / unionlap.sum()

        dices.append(dice)
        ious.append(iou)
        target_cell_label.append(true_id)
        target_cell_pixels.append((true==true_id).sum())

    return target_cell_label,target_cell_pixels, ious,dices


def seg_score():
    # configuration
    config = dict(net='SwinUNETR',  # you can change the model to see different outcome
                  source_folder=r'E:\CellAtlas\CellAtlas-dev_topology_loss\OutputData',
                  gt_folder=r'E:\CellAtlas\CellAtlas-dev_topology_loss\OutputData\groundTruth',  # ground truth cell segmentation
                  input_nuc=True,  # whether to use nucleus information
                  size=5  # to change the size of the seed, bigger size --> less seeds
                  )

    if config['input_nuc'] is True:

        memb_files = glob.glob(os.path.join(config['source_folder'], config['net'], "SegMemb", "*nii.gz"))
        target_files_path = os.path.join(config['source_folder'], config['net'], "with_nucleus_cell_folder")

        if not os.path.exists(target_files_path):
            os.makedirs(target_files_path)
        gt_files_path = config['gt_folder']

        start_time = time.time()

        for file in tqdm(memb_files, total=len(memb_files)):
            instance_segmentation_with_nucleus(file, target_files_path, config['size'])

        remap_labels_for_seg(target_files_path, gt_files_path)

        _, pd_embryo_avg_score = iou_dicescore_evaluate(target_files_path, gt_files_path)

        save_file = os.path.join(config['source_folder'], "Comparsion",
                                 config['net'] + "(with_nucleus)_avg_evaluation.csv")
        if os.path.isfile(save_file):
            open(save_file, "w").close()

        pd_embryo_avg_score.to_csv(save_file, index=False)

        end_time = time.time()
        print("It costs %d minutes to segment and evaluate" % ((end_time - start_time) / 60))


    elif config['input_nuc'] is False:

        memb_files = glob.glob(os.path.join(config['source_folder'], config['net'], "SegMemb", "*nii.gz"))
        target_files_path = os.path.join(config['source_folder'], config['net'], "without_nucleus_cell_folder")

        if not os.path.exists(target_files_path):
            os.makedirs(target_files_path)
        gt_files_path = config['gt_folder']

        start_time = time.time()

        for file in tqdm(memb_files, total=len(memb_files)):
            instance_segmentation_without_nucleus(file, target_files_path, config['size'])

        remap_labels_for_seg(target_files_path, gt_files_path)

        _, pd_embryo_avg_score = iou_dicescore_evaluate(target_files_path, gt_files_path)

        save_file = os.path.join(config['source_folder'], "Comparsion",
                                 config['net'] + "(without_nucleus)_avg_evaluation.csv")
        if os.path.isfile(save_file):
            open(save_file, "w").close()

        pd_embryo_avg_score.to_csv(save_file, index=False)

        end_time = time.time()
        print("It costs %d minutes to segment and evaluate" % ((end_time - start_time) / 60))


def watershed_processing(binary_nuc):
    binary_nuc = binary_nuc.astype(int)
    distance = ndi.distance_transform_edt(binary_nuc)
    max_coords = peak_local_max(distance, labels=binary_nuc, footprint=np.ones((3, 3, 3)))#best dice score
    local_maxima = np.zeros_like(binary_nuc, dtype=bool)
    local_maxima[tuple(max_coords.T)] = True
    markers = ndi.label(local_maxima)[0]
    labels = watershed(-distance, markers, mask=binary_nuc)

    # find center points in watershell
    unique_values = np.unique(labels)
    center_coords = []

    for value in unique_values:
        coords = np.array(np.where(labels == value)).T
        center_coord = np.mean(coords, axis=0)
        center_coords.append((int(center_coord[0]), int(center_coord[1]), int(center_coord[2])))

    int_mask = np.zeros_like(labels, dtype=np.uint8)

    for coord in center_coords:
        int_mask[coord[0], coord[1], coord[2]] = labels[coord[0], coord[1], coord[2]]

    return int_mask

def dist_processing(binary_nuc):
    binary_nuc = binary_nuc.astype(int)
    distance = ndi.distance_transform_edt(binary_nuc)
    return distance




def convert_binary():
    input_directory = r'E:\3DSegEvaluation\StardistGt/'
    output_diectory = r'E:\3DSegEvaluation\StarbinaryGt/'
    files = os.listdir(input_directory)
    for file in files:
        filename_parts = file.split('_')
        tp = int(filename_parts[1])
        input_name = input_directory + file
        int_nuc = nib_load(input_name)
        binary_nuc = np.where(int_nuc > 0, 1, 0)  # for stardist
        save_name = output_diectory + filename_parts[0] + '_' + filename_parts[1] + '_' + 'segNuc.nii.gz'
        print('processing: ', save_name)
        nib_save(binary_nuc, save_name)


if __name__ == '__main__':
    seg_score()
