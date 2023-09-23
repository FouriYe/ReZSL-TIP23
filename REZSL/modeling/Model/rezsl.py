import torch
import torch.nn as nn
import torch.nn.functional as F

class ReZSL(nn.Module):
    def __init__(self, p, p2, att_dim, train_class_num, test_class_num, RegNorm, RegType, WeightType, device=None):
        super(ReZSL, self).__init__()
        self.att_dim = att_dim
        self.train_class_num = train_class_num
        self.test_class_num = test_class_num
        self.p = p
        self.p2 = p2
        self.RegNorm = RegNorm
        self.RegType = RegType
        self.WeightType = WeightType
        # training stage
        self.device = device
        self.running_offset_Matrix = torch.zeros(self.train_class_num, self.att_dim, requires_grad=False).to(self.device)
        self.running_weights_Matrix = torch.ones(self.train_class_num, self.att_dim, requires_grad=False).to(self.device)
        self.min_value = torch.zeros(self.att_dim, requires_grad=False).to(self.device)
        self.mean_value = torch.zeros(self.att_dim, requires_grad=False).to(self.device)
        # testing stage
        self.test_cls_offset_sum = torch.zeros(self.test_class_num, self.att_dim, requires_grad=False).to(self.device)
        self.test_cls_count = torch.zeros(self.test_class_num, dtype=torch.int64, requires_grad=False).to(self.device)
        self.test_cls_offset_mean = torch.zeros(self.test_class_num, self.att_dim, requires_grad=False).to(self.device)
        self.test_mean_value = torch.zeros(self.att_dim, requires_grad=False).to(self.device)

    def updateWeightsMatrix(self, batch_pred, batch_truth, batch_label):
        if self.WeightType=="in_batch":
            self.updateWeightsMatrix_inBatch(batch_pred, batch_truth, batch_label)
        elif self.WeightType=="cross_batch":
            self.updateWeightsMatrix_inBatch(batch_pred, batch_truth, batch_label)

    def updateWeightsMatrix_inBatch(self, batch_pred, batch_truth, batch_label):
        self.running_offset_Matrix = self.arrangeTrainOffset(batch_pred, batch_truth, batch_label)

        self.mean_value = torch.mean(self.running_offset_Matrix, dim=0)
        self.running_weights_Matrix = self._updateWeightsMatrix(self.running_offset_Matrix)

        return self.running_offset_Matrix, self.running_weights_Matrix

    def updateWeightsMatrix_crossBatch(self, batch_pred, batch_truth, batch_label):
        previous_offset_Matrix = self.running_offset_Matrix
        self.running_offset_Matrix = self.arrangeTrainOffset(batch_pred, batch_truth, batch_label)

        unchanged_index = torch.nonzero(self.running_offset_Matrix == 0.0, as_tuple=True)
        self.running_offset_Matrix = self.running_offset_Matrix.index_put(unchanged_index, previous_offset_Matrix[unchanged_index])

        self.mean_value = torch.mean(self.running_offset_Matrix, dim=0)
        self.running_weights_Matrix = self._updateWeightsMatrix(self.running_offset_Matrix)

        return self.running_offset_Matrix, self.running_weights_Matrix

    def _updateWeightsMatrix(self, input, eps = 1e-10):
        """
        input: 2-D tensor [class_num, feature_dim (semantic feature dim indeed)]
        scale_factor:  2-D tensor [class_num, feature_dim]
        compute weights matrix by: (cur/min)**p
        """
        data = input.T  # data: transposition input [feature_dim, class_num]
        feature_dim, class_num = data.shape
        mask = torch.gt(data, 0.0)
        # intra-semantic (cross-class)
        WeightsMatrix1 = torch.ones(feature_dim, class_num, requires_grad=False).to(self.device)  # needed to be transposed
        for i in range(feature_dim):
            selected_row = torch.masked_select(data[i], mask[i])
            if not selected_row.numel() == 0:
                self.min_value[i] = torch.min(selected_row)
                WeightsMatrix1[i] = torch.log(data[i] / (self.min_value[i]))+1.0
        WeightsMatrix1 = WeightsMatrix1 ** self.p
        WeightsMatrix1 = WeightsMatrix1.masked_fill(~mask, 1.0)
        WeightsMatrix1 = WeightsMatrix1.T # transposed
        # inter-semantic (cross-semantic)
        WeightsMatrix2 = torch.ones(class_num, feature_dim, requires_grad=False).to(self.device)
        for i in range(class_num):
            selected_col = torch.masked_select(data.T[i], mask.T[i])
            if not selected_col.numel() == 0:
                min_value = torch.min(selected_col)
                WeightsMatrix2[i] = torch.log(data.T[i] / (min_value)) + 1.0
        WeightsMatrix2 = WeightsMatrix2 ** self.p2
        WeightsMatrix2 = WeightsMatrix2.masked_fill(~(mask.T), 1.0)

        WeightsMatrix = WeightsMatrix1*WeightsMatrix2
        return WeightsMatrix

    def arrangeTrainOffset(self, batch_pred, batch_truth, batch_label):
        """
        sort out batch offset into class-wised offset, then average them
        input: batch_pred[batch_size, att_dim], batch_truth[batch_size, att_dim], batch_label[batch_size]
        return: cls_offset_mean [class_num, att_dim]
        """
        # initiating
        batch_size = batch_pred.shape[0]
        # norm pred and semantic to identical vector
        if self.RegNorm == True:
            batch_pred_norm = batch_pred.norm(p=2, dim=1, keepdim=True).expand_as(batch_pred)
            batch_pred_ = batch_pred / (batch_pred_norm + 1e-10)
            batch_truth_norm = batch_truth.norm(p=2, dim=1, keepdim=True).expand_as(batch_truth)
            batch_truth_ = batch_truth / (batch_truth_norm + 1e-10)
        else:
            batch_pred_ = batch_pred
            batch_truth_ = batch_truth

        if self.RegType == "MSE" or self.RegType == "BMC":
            batch_offset_ori = (batch_pred_ - batch_truth_) ** 2
        elif self.RegType == "RMSE":
            batch_offset_ori = (batch_pred_ - batch_truth_) ** 2
            batch_offset_ori = torch.sqrt(batch_offset_ori + 1e-10)
        elif self.RegType == "MAE":
            batch_offset_ori = torch.abs(inputs - targets)
        else:
            raise TypeError(self.RegType + "is not implemented")

        cls_offset_sum = torch.zeros(self.train_class_num, self.att_dim, requires_grad=False).to(self.device)
        cls_count = torch.zeros(self.train_class_num, dtype=torch.int64, requires_grad=False).to(self.device)
        cls_offset_mean = torch.zeros(self.train_class_num, self.att_dim, requires_grad=False).to(self.device)
        #calculating
        for i in range(batch_size):
            cls_offset_sum[batch_label[i]] = cls_offset_sum[batch_label[i]] + batch_offset_ori[i]
            cls_count[batch_label[i]] = cls_count[batch_label[i]] + 1
        #masking and averaging
        for i in range(self.train_class_num):
            if cls_count[i] > 0:
                cls_offset_mean[i] = cls_offset_sum[i]/cls_count[i]

        return cls_offset_mean

    def arrangeTestOffset(self, batch_pred, batch_truth, batch_label):
        """
        sort out batch offset into class-wised offset (without averaging)
        input: batch_pred [batch_size, att_dim], batch_truth [batch_size, att_dim], batch_label [batch_size]
        """
        # initiating
        batch_size = batch_pred.shape[0]
        if self.RegNorm == True:
            batch_pred_norm = batch_pred.norm(p=2, dim=1, keepdim=True).expand_as(batch_pred)
            batch_pred_ = batch_pred / (batch_pred_norm + 1e-10)
            batch_truth_norm = batch_truth.norm(p=2, dim=1, keepdim=True).expand_as(batch_truth)
            batch_truth_ = batch_truth / (batch_truth_norm + 1e-10)
        else:
            batch_pred_ = batch_pred
            batch_truth_ = batch_truth

        if self.RegType == "MSE" or self.RegType == "BMC":
            batch_offset_ori = (batch_pred_ - batch_truth_) ** 2
        elif self.RegType == "RMSE":
            batch_offset_ori = torch.sqrt((batch_pred_ - batch_truth_) ** 2 + 1e-12)
        elif self.RegType == "MAE":
            batch_offset_ori = torch.abs(batch_pred_ - batch_truth_)
        else:
            raise TypeError(self.RegType + "is not implemented")

        # calculating
        for i in range(batch_size):
            self.test_cls_offset_sum[batch_label[i]] = self.test_cls_offset_sum[batch_label[i]] + batch_offset_ori[i]
            self.test_cls_count[batch_label[i]] = self.test_cls_count[batch_label[i]] + 1

    def averageTestOffset(self):
        # averaging
        for i in range(self.test_class_num):
            if self.test_cls_count[i] > 0:
                self.test_cls_offset_mean[i] = self.test_cls_offset_sum[i] / self.test_cls_count[i]
        self.test_mean_value = torch.mean(self.test_cls_offset_mean, dim=0)
        return self.test_cls_offset_mean, self.test_mean_value

    def afterTest(self):
        self.test_cls_offset_sum = torch.zeros(self.test_class_num, self.att_dim, requires_grad=False).to(self.device)
        self.test_cls_count = torch.zeros(self.test_class_num, dtype=torch.int64, requires_grad=False).to(self.device)
        self.test_cls_offset_mean = torch.zeros(self.test_class_num, self.att_dim, requires_grad=False).to(self.device)
        self.test_mean_value = torch.zeros(self.att_dim, requires_grad=False).to(self.device)

    def getWeights(self, batch_size, feature_dim, label_v):
        """
        Input:
        batch_size = n
        feature_dim = s
        label_v = [n]
        Output:
        weights: [n, s]
        """

        weight_scale = batch_size*feature_dim
        weights = torch.ones((batch_size,feature_dim)).to(self.device)

        for i in range(batch_size):
            weights[i] = self.running_weights_Matrix[label_v[i]]
        return weights

def computeCoefficient(batch_pred,batch_att):
    """
    Input:
    batch_pred: [n,s]
    batch_att: [n,s]
    Output:
    coefficient: [s]
    """
    batch_pred_norm = batch_pred.norm(p=2, dim=1, keepdim=True).expand_as(batch_pred)
    batch_pred_normalized = batch_pred / (batch_pred_norm + 1e-10)
    batch_att_norm = batch_att.norm(p=2, dim=1, keepdim=True).expand_as(batch_att)
    batch_att_normalized = batch_att / (batch_att_norm + 1e-10)
    weight_scale = 312
    offset = torch.abs(batch_pred_normalized - batch_att_normalized)
    offset_mean = offset.mean(dim=0,keepdim=True)

    offset_mean_log = torch.log(offset_mean)
    coefficient = F.softmax(offset_mean_log) * weight_scale
    coefficient = torch.squeeze(coefficient,dim=0)

    return coefficient

def recordError(batch_pred, batch_att, offset_recorder, num_recorrder):
    """
    Input:
    batch_pred: [n,s]
    batch_att: [n,s]
    offset_recorder: [s,k]
    num_recorrder: int
    Output:
    offset_recorder: [s,k+1]
    num_recorrder: int
    """
    batch_pred_norm = batch_pred.norm(p=2, dim=1, keepdim=True).expand_as(batch_pred)
    batch_pred_normalized = batch_pred / (batch_pred_norm + 1e-10)
    batch_att_norm = batch_att.norm(p=2, dim=1, keepdim=True).expand_as(batch_att)
    batch_att_normalized = batch_att / (batch_att_norm + 1e-10)
    offset = torch.abs(batch_pred_normalized - batch_att_normalized)
    offset_sum = offset.sum(dim=0, keepdim=True) # [s,1]
    # offset = torch.log(offset)

    offset_recorder = torch.cat((offset_recorder, offset_sum), dim=0)  # [s,k+1]
    num_recorrder = num_recorrder+batch_att.shape[0]

    return offset_recorder, num_recorrder
