#
'''
Author: Hao Wei
Date: 2019/06/03
Function:
'''
import torch
import numpy as np
import config

class Metric():
    def __init__(self,reduction = 'mean'):
        self.reduciton = reduction

    def set_input(self,y_true, y_pred):
        '''

        :param y_true: torch.Tensor [channel, h, w] or [batch, channel, h, w] in range of [0,num_class-1]
        :param y_pred: torch.Tensor [channel, h, w] or [batch,channel, h, w] in range of [0,num_class-1]
        :return:
        '''
        self.y_true_batch = None # for the batch-format y_true
        self.y_pred_batch = None # for the batch-format y_pred
        self.y_true_case = None # for the case-format y_true
        self.y_pred_case = None # for the case-format y_pred
        self.shape = y_true.shape
        self.num_class = config.num_class

        if len(self.shape) == 4: # the batch-format
            assert y_true.shape == y_pred.shape, 'the shape of y_true and y_pred should be same!'
            self.y_true_batch = y_true.clone().detach().cpu().numpy()
            self.y_pred_batch = y_pred.clone().detach().cpu().numpy()

        if len(self.shape) == 3: # the no batch-fromat
            assert y_true.shape == y_pred.shape, 'the shape of y_true and y_pred should be same!'
            self.y_true_case = y_true.clone().detach().cpu().numpy()
            self.y_pred_case = y_pred.clone().detach().cpu().numpy()

    def dice_for_case(self):

        # out = torch.rand(self.num_class)
        out = np.zeros((self.num_class))
        assert len(self.y_pred_case.shape) == 3, 'the input for dice_for_batch should has 3 dims'

        # for organ_index in range(self.num_class):
        #     pred_organ = torch.zeros_like(self.y_pred_case)
        #     target_organ = torch.zeros_like(self.y_true_case)
        #
        #     pred_organ[self.y_pred_case == organ_index] = 1
        #     target_organ[self.y_true_case == organ_index] = 1
        #     if target_organ.sum() == 0:
        #         out[organ_index] = -1
        #         return None  # denotes that there isn't this class.
        #     else:
        #         dice = (2 * pred_organ * target_organ).sum() / (pred_organ.sum() + target_organ.sum())
        #         out[organ_index] = dice

        try:
            # Compute tumor+kidney Dice
            tk_pd = np.greater(self.y_pred_case, 0)
            tk_gt = np.greater(self.y_true_case, 0)
            tk_dice = 2 * np.logical_and(tk_pd, tk_gt).sum() / (
                    tk_pd.sum() + tk_gt.sum() + 1e-5
            )
        except ZeroDivisionError:
            return 0.0, 0.0

        try:
            # Compute tumor Dice
            tu_pd = np.greater(self.y_pred_case, 1)
            tu_gt = np.greater(self.y_true_case, 1)
            tu_dice = 2 * np.logical_and(tu_pd, tu_gt).sum() / (
                    tu_pd.sum() + tu_gt.sum() + 1e-5
            )
        except ZeroDivisionError:
            return tk_dice, 0.0
        out[0] = tk_dice
        out[1] = tu_dice
        return out

    def dice_for_batch(self):
        assert len(self.shape) == 4, 'the input for dice_for_batch should has 4 dims'
        # out = torch.rand(self.shape[0], self.num_class)
        out = np.zeros((self.shape[0], self.num_class))

        for batch_index in range(self.shape[0]):
            self.y_true_case = self.y_true_batch[batch_index]
            self.y_pred_case = self.y_pred_batch[batch_index]
            out[batch_index] = self.dice_for_case()

        return out


if __name__ == '__main__':
    num_class = 3
    hight, width, = 100, 100
    batch = 4
    # define the object of Metric class
    metricer = Metric()
    # for testing the dice_for_case
    prediciton = torch.randint(low=0, high=num_class, size=(num_class, hight, width),dtype=torch.float)
    label = torch.randint(low=0, high=num_class, size=(num_class, hight, width),dtype=torch.float)
    metricer.set_input(label, prediciton)
    print('test for the dice_for_case')
    print('dice: ', metricer.dice_for_case())

    prediciton = torch.randint(low=0, high=num_class, size=(batch, num_class, hight, width),dtype=torch.float)
    label = torch.randint(low=0, high=num_class, size=(batch, num_class, hight, width),dtype=torch.float)
    metricer.set_input(label, prediciton)
    print('test for the dice_for_batch')
    print('dice: ', metricer.dice_for_batch())