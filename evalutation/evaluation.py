from utils import load_segmentation

import numpy as np
import nibabel as nib
import glob
import re

def evaluate(file_gt, predictions, object=None):

    # Handle case of softmax output
    if len(predictions.shape) == 4:
        predictions = np.argmax(predictions, axis=-1)

    # Check predictions for type and dimensions
    if not isinstance(predictions, (np.ndarray, nib.Nifti1Image)):
        raise ValueError("Predictions must by a numpy array or Nifti1Image")
    if isinstance(predictions, nib.Nifti1Image):
        predictions = predictions.get_data()

    if not np.issubdtype(predictions.dtype, np.integer):
        predictions = np.round(predictions)
    predictions = predictions.astype(np.uint8)
    # predictions = predictions.transpose(2,0,1)
    # predictions = predictions.swapaxes(1,2)
    # Load ground truth segmentation
    gt = load_segmentation(file_gt).get_data()

    # Make sure shape agrees with case
    if not predictions.shape == gt.shape:
        raise ValueError(
            ("Predictions for case {} have shape {} "
            "which do not match ground truth shape of {}").format(
                file_gt, predictions.shape, gt.shape
            )
        )
    if object == 'tumor':
        tu_pd = np.greater(predictions, 0)
        tu_gt = np.greater(gt, 1)
        tu_dice = 2*np.logical_and(tu_pd, tu_gt).sum()/(
            tu_pd.sum() + tu_gt.sum()
        )
        tk_dice = 0.0
    elif object == 'kidney':
        tk_pd = np.greater(predictions, 0)
        tk_gt = np.greater(gt, 0)
        tk_dice = 2*np.logical_and(tk_pd, tk_gt).sum()/(
            tk_pd.sum() + tk_gt.sum()
        )
        tu_dice = 0.0
    else:
        try:
            # Compute tumor+kidney Dice
            tk_pd = np.greater(predictions, 0)
            tk_gt = np.greater(gt, 0)
            tk_dice = 2*np.logical_and(tk_pd, tk_gt).sum()/(
                tk_pd.sum() + tk_gt.sum()
            )
        except ZeroDivisionError:
            return 0.0, 0.0

        try:
            # Compute tumor Dice
            tu_pd = np.greater(predictions, 1)
            tu_gt = np.greater(gt, 1)
            tu_dice = 2*np.logical_and(tu_pd, tu_gt).sum()/(
                tu_pd.sum() + tu_gt.sum()
            )
        except ZeroDivisionError:
            return tk_dice, 0.0
    tk_gt = np.greater(gt, 0)
    tu_gt = np.greater(gt, 1)
    ratio = np.sum(tu_gt) / np.sum(tk_gt)
    return tk_dice, tu_dice, round(ratio,3)

if __name__ == '__main__':
    root_gt = '/data/weihao/pre-KiTS-3mm/val/GT/'
    root_pred = '/data/weihao/pre-KiTS-3mm/predictions_multistage_data_aug/'
    files_gt = glob.glob(root_gt+'*.nii')
    files_gt = [file for file in files_gt if 'back' not in file] # filtered the back case
    files_gt.sort()

    # f = open('/data/weihao/pre-KiTS-3mm/val_4.txt')
    # files_gt = f.readlines()
    # files_gt = [file[:-1] for file in files_gt]
    # test_list = [file for file in test_list if 'back' not in file]
    # files_gt.sort()

    files_pred = glob.glob(root_pred+'*.nii')
    files_pred.sort()
    assert len(files_pred) == len(files_gt), 'the length of file list in pred and GT shoule be same'

    kidney_dice = []
    tumor_dice = []
    f = open(root_pred + 'statistics_for_dice.txt', 'w')
    for index in range(len(files_pred)):
        assert re.findall('-\d+?\.nii',files_pred[0])[0] == re.findall('-\d+?\.nii',files_gt[0])[0], \
            'the corresponding file of pred and GT should be same'
        predictions = load_segmentation(files_pred[index])
        label_path = files_gt[index].replace('CT', 'GT')
        label_path = label_path.replace('img', 'label')
        tk_dice, tu_dice, ratio = evaluate(label_path, predictions)
        tk_dice = round(tk_dice, 3)
        tu_dice = round(tu_dice, 3)
        kidney_dice.append(tk_dice)
        tumor_dice.append(tu_dice)
        f.write(files_pred[index] + '  {}  {} {} \n'.format(tk_dice, tu_dice, ratio))
        print('the {} process completed! the dice are {}  {}'.format(files_pred[index], tk_dice, tu_dice))

    mean_kidney_dice = round(np.array(kidney_dice).mean(),3)
    mean_tumor_dice = round(np.array(tumor_dice).mean(),3)
    print('the mean dice are {}  {}'.format(mean_kidney_dice, mean_tumor_dice))
    f.write('mean kidney dice: {} tumor dice: {} \n'.format(mean_kidney_dice,mean_tumor_dice))
    f.close()