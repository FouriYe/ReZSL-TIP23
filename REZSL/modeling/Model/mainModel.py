import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class GEMNet(nn.Module):
    def __init__(self, res101, ft_flag, img_size, c, w, h,
                 attritube_num, cls_num, ucls_num, attr_group, w2v,
                 scale=20.0, device=None):
        super(GEMNet, self).__init__()
        self.device = device

        self.name = "GEMNet"

        self.img_size = img_size
        # self.prototype_shape = prototype_shape
        self.attritube_num = attritube_num

        self.feat_channel = c
        self.feat_w = w
        self.feat_h = h

        self.ucls_num = ucls_num
        self.scls_num = cls_num - ucls_num
        self.attr_group = attr_group

        self.w2v_att = torch.from_numpy(w2v).float().to(self.device)  # 312 * 300
        assert self.w2v_att.shape[0] == self.attritube_num

        if scale <= 0:
            self.scale = nn.Parameter(torch.ones(1) * 20.0)
        else:
            self.scale = nn.Parameter(torch.tensor(scale), requires_grad=False)

        self.backbone = res101
        self.ft_flag = ft_flag
        self.check_fine_tune()

        self.W = nn.Parameter(nn.init.normal_(torch.empty(self.w2v_att.shape[1], self.feat_channel)),
                              requires_grad=True)  # 300 * 2048

        self.V = nn.Sequential(nn.Linear(self.feat_channel, self.attritube_num))  # V, S = [2048,4096]

    def forward(self, x, att=None, label=None, support_att=None):

        feat = self.conv_features(x)  # N， 2048， 14， 14

        v2s = self.base_module(feat, support_att)  # N, d

        part_feat, atten_map, atten_v2s, query = self.attentionModule(feat)

        if not self.training:
            return v2s

        return v2s, atten_v2s, atten_map, query

    def base_module(self, x, seen_att):

        N, C, W, H = x.shape
        global_feat = F.avg_pool2d(x, kernel_size=(W, H))
        global_feat = global_feat.view(N, C)
        v2s = self.V(global_feat)

        return v2s

    def attentionModule(self, x):

        N, C, W, H = x.shape
        x = x.reshape(N, C, W * H)  # N, V, r=WH

        self.w2v_att=self.w2v_att.to(self.W.device)

        query = torch.einsum('lw,wv->lv', self.w2v_att, self.W) # L * V

        atten_map = torch.einsum('lv,bvr->blr', query, x) # batch * L * r

        atten_map = F.softmax(atten_map, -1)

        x = x.transpose(2, 1) # batch, WH=r, V
        part_feat = torch.einsum('blr,brv->blv', atten_map, x) # batch * L * V
        part_feat = F.normalize(part_feat, dim=-1)

        atten_map = atten_map.view(N, -1, W, H)
        atten_v2s = F.max_pool2d(atten_map, kernel_size=(W,H))
        atten_v2s = atten_v2s.view(N, -1)

        return part_feat, atten_map, atten_v2s, query

    def conv_features(self, x):
        '''
        the feature input to prototype layer
        '''
        x = self.backbone(x)
        return x

    def cosine_dis(self, pred_att, support_att):
        pred_att_norm = torch.norm(pred_att, p=2, dim=1).unsqueeze(1).expand_as(pred_att)
        pred_att_normalized = pred_att.div(pred_att_norm + 1e-10)
        support_att_norm = torch.norm(support_att, p=2, dim=1).unsqueeze(1).expand_as(support_att)
        support_att_normalized = support_att.div(support_att_norm + 1e-10)
        cos_dist = torch.einsum('bd,nd->bn', pred_att_normalized, support_att_normalized)
        score = cos_dist * self.scale  # B, cls_num
        return score, cos_dist

    def check_fine_tune(self):
        if self.ft_flag:
            for p in self.backbone.parameters():
                p.requires_grad = True
        else:
            for p in self.backbone.parameters():
                p.requires_grad = False


class BasicNet(nn.Module):
    def __init__(self, backbone, backbone_type, ft_flag, img_size, hid_dim, c, w, h,
                 attritube_num, cls_num, ucls_num,
                 scale=20.0, device=None):

        super(BasicNet, self).__init__()
        self.device = device
        self.name = "BasicNet"
        self.img_size = img_size
        # self.prototype_shape = prototype_shape
        self.attritube_num = attritube_num

        self.feat_channel = c
        self.feat_w = w
        self.feat_h = h

        self.ucls_num = ucls_num
        self.scls_num = cls_num - ucls_num

        if scale <= 0:
            self.scale = torch.ones(1) * 20.0
        else:
            self.scale = torch.tensor(scale)
        self.backbone_type = backbone_type
        self.backbone = backbone  # requires_grad = True
        self.ft_flag = ft_flag
        self.check_fine_tune()

        self.hid_dim = hid_dim
        if self.hid_dim > 0:
            self.v2hid = nn.Sequential(nn.Linear(self.feat_channel, self.hid_dim))  # C, H = [2048,4096]
            self.LeakyReLU1 = torch.nn.LeakyReLU(negative_slope=0.1, inplace=True)
            self.hid2s = nn.Sequential(nn.Linear(self.hid_dim, self.attritube_num))  # V, S = [2048,4096]
        else:
            self.v2s = nn.Sequential(nn.Linear(self.feat_channel, self.attritube_num))  # V, S = [2048,4096]

    def forward(self, x, support_att):
        if self.backbone_type == "resnet":
            vis_feat = self.conv_features(x)  # N， 2048， 14， 14
            v2s = self.base_module(vis_feat)  # N, d
        else:
            global_feat, patch_feat = self.conv_features(x)
            B, C = global_feat.shape
            patch_feat = patch_feat.permute(0, 2, 1)
            feat = torch.cat([global_feat.view(B, C,-1), patch_feat], dim=2)  # [B, C, N+1]
            v2s = self.base_module(feat)  # B,312
        return v2s

    def base_module(self, x):
        # support_att[class_dim, att_dim]
        if len(x.shape) == 3:
            B, C, N = x.shape
            global_feat = F.avg_pool1d(x, kernel_size=(N))
        else:
            B, C, W, H = x.shape
            global_feat = F.avg_pool2d(x, kernel_size=(W, H))
        global_feat = global_feat.view(B, C) # N, V
        if self.hid_dim > 0:
            v2s = self.hid2s(self.LeakyReLU1(self.v2hid(global_feat)))
        else:
            v2s = self.v2s(global_feat)

        return v2s

    def conv_features(self, x):
        '''
        the feature input to prototype layer
        '''
        x = self.backbone(x)
        return x

    def cosine_dis(self, pred_att, support_att, norm=True):
        """
        Input:
        pred_att: [n,s]
        support_att: [c,s]
        weights: [n,s] else None
        Output:
        score: [n,c]
        cos_dist: [n,c]
        """

        if norm:
            pred_att_norm = torch.norm(pred_att, p=2, dim=1).unsqueeze(1).expand_as(pred_att)
            pred_att_normalized = pred_att.div(pred_att_norm + 1e-10)
            support_att_norm = torch.norm(support_att, p=2, dim=1).unsqueeze(1).expand_as(support_att)
            support_att_normalized = support_att.div(support_att_norm + 1e-10)
            cos_dist = torch.einsum('bs,cs->bc', pred_att_normalized, support_att_normalized)
            score = cos_dist * self.scale
            return score, cos_dist
        else:
            cos_dist = torch.einsum('ns,cs->nc', pred_att, support_att)
            score = cos_dist * self.scale
            return score, cos_dist

    def check_fine_tune(self):
        if self.ft_flag:
            for p in self.backbone.parameters():
                p.requires_grad = True
        else:
            for p in self.backbone.parameters():
                p.requires_grad = False


class AttentionNet(nn.Module):
    def __init__(self, backbone, backbone_type, ft_flag, img_size, hid_dim, c, w, h,
                 attritube_num, cls_num, ucls_num, attr_group, w2v,
                 scale=20.0, device=None):
        super(AttentionNet, self).__init__()

        # self.prototype_shape = prototype_shape
        self.device = device

        self.name = "AttentionNet"

        self.img_size = img_size
        # self.prototype_shape = prototype_shape
        self.attritube_num = attritube_num

        self.feat_channel = c
        self.feat_w = w
        self.feat_h = h
        self.feat_n = w * h

        self.ucls_num = ucls_num
        self.scls_num = cls_num - ucls_num
        self.attr_group = attr_group
        # global branch

        self.w2v_att = torch.from_numpy(w2v).float().to(self.device)  # 312 * 300
        assert self.w2v_att.shape[0] == self.attritube_num
        _, self.w2v_length = self.w2v_att.shape

        if scale <= 0:
            self.scale = torch.ones(1) * 20.0
        else:
            self.scale = torch.tensor(scale)
        if attritube_num == 85 and backbone_type=='resnet':
            self.W = nn.Parameter(nn.init.normal_(torch.empty(self.w2v_att.shape[1], self.feat_channel)),
                                  requires_grad=True)  # 300 * 2048
            self.V = nn.Parameter(nn.init.normal_(torch.empty(self.feat_channel, self.attritube_num)),
                                  requires_grad=True)
            self.ratio = 0.125
        else:
            self.ratio = 1.0

        self.backbone_type = backbone_type
        self.backbone = backbone  # requires_grad = True

        self.ft_flag = ft_flag
        self.check_fine_tune()

        # local branch
        if attritube_num == 85 and backbone_type=='resnet':
            self.med_dim = 300 # 1024
            self.QueryW = nn.Sequential(nn.Linear(self.w2v_length, self.med_dim)) # L,M = 300,1024
            self.KeyW = nn.Sequential(nn.Linear(self.feat_channel, self.med_dim)) # C,M = 2048,1024
            self.ValueW = nn.Sequential(nn.Linear(self.feat_channel, self.med_dim))  # C,M = 2048,1024
            self.W_o = nn.Sequential(nn.Linear(self.med_dim, self.feat_channel)) # M,C = 1024,2048
        else:
            self.med_dim = 1024  # 1024
            self.QueryW = nn.Linear(self.w2v_length, self.med_dim)  # L,M = 300,1024
            self.KeyW = nn.Linear(self.feat_channel, self.med_dim)  # C,M = 2048,1024
            self.ValueW = nn.Linear(self.feat_channel, self.med_dim)  # C,M = 2048,1024
            self.W_o = nn.Linear(self.med_dim, self.feat_channel)  # M,C = 1024,2048

        # hidden layer or not
        self.hid_dim = hid_dim
        if self.hid_dim == 0:
            self.V_att_final_branch = nn.Parameter(nn.init.normal_(torch.empty(self.attritube_num, self.feat_channel)),
                                                   requires_grad=True)  # S, C
        else:
            self.V_att_hidden_branch = nn.Sequential(nn.Linear(self.feat_channel, self.hid_dim))  # H, C
            self.V_att_final_branch = nn.Parameter(nn.init.normal_(torch.empty(self.attritube_num, self.hid_dim)),
                                                   requires_grad=True)  # S, H

    def forward(self, x, label_att=None, label=None, support_att=None, getAttention = False):
        if self.backbone_type=="resnet":
            feat = self.conv_features(x)  # B， 2048， 14， 14
            if getAttention:
                v2s, attentionMap = self.res_attention_module(feat, support_att, getAttention)  # B, 312
                return v2s, attentionMap
            else:
                v2s = self.res_attention_module(feat, support_att)  # B,312
                return v2s
        else:
            global_feat, patch_feat = self.conv_features(x)  # B, 2048, 14, 14
            patch_feat = patch_feat.permute(0, 2, 1)
            B,C = global_feat.shape
            if getAttention:
                v2s, attentionMap = self.vit_attention_module(global_feat.view(B,C,1), patch_feat, support_att, getAttention)  # B, 312
                return v2s, attentionMap
            else:
                v2s = self.vit_attention_module(global_feat.view(B,C,1), patch_feat, support_att)  # B,312
                return v2s

    def vit_attention_module(self, global_feat, patch_feat, s, getAttention = False):
        """
        global_feat: [B, C, 1]
        patch_feat: [B, C, N=W*H]
        """
        B, C, N = patch_feat.shape
        W = H = int(N ** 0.5)
        S, L = self.w2v_att.shape
        M = self.med_dim

        feat = torch.cat([global_feat,patch_feat],dim=2) # [B, C, N+1]

        # attention feature
        w2v_att = self.w2v_att.to(torch.cuda.current_device())
        query = self.QueryW(w2v_att) # [S, L]*[L,M] -> [S,M]
        query_batch = query.unsqueeze(0).repeat(B, 1, 1)  # [S,M] -> [1,S,M] -> [B,S,M]
        key = self.KeyW(feat.permute(0, 2, 1))  # [B, C, N+1] -> [B, N+1, C] -> [B, N+1, M]
        value = self.ValueW(feat.permute(0, 2, 1))  # [B, C, N+1] -> [B, N+1, C] -> [B, N+1, M]

        attention = F.softmax(torch.matmul(query_batch, key.permute(0, 2, 1)),
                              dim=2)  # [B,S,M],[B,N,M] -> [B,S,M],[B,M,N] -> [B, S, N]
        attented_feat = torch.matmul(attention, value)  # [B, S, N] * [B, N, M] -> [B,S,M]
        attented_feat_o = self.W_o(attented_feat)  # [B,S,M] -> [B,S,C]

        feat_pool = F.avg_pool1d(feat, kernel_size=(N+1))  # B, C
        feat_reshape_repeat = feat_pool.view(B, 1, -1).expand(B, self.attritube_num, C)  # B, S, C
        attented_feat_final = feat_reshape_repeat + self.ratio * attented_feat_o  # [B,S,C]

        # visual to semantic
        if self.hid_dim == 0:
            v2s = torch.einsum('BSC,SC->BS', attented_feat_final, self.V_att_final_branch)  # [B,312,2048] * [312, 2048] -> [B,312,2048] -> [B,312]
        else:
            attented_feat_hid = self.V_att_hidden_branch(attented_feat_final) # [B,312,2048] -> [B,312,4096]
            v2s = torch.einsum('BSH,SH->BS', attented_feat_hid, self.V_att_final_branch)
        if getAttention:
            return v2s, attention
        else:
            return v2s

    def res_attention_module(self, feat, s, getAttention = False):
        """
        feat: [B, C, W, H]
        """
        B, C, W, H = feat.shape
        N = W * H
        S, L = self.w2v_att.shape
        M = self.med_dim
        W = H = int(N ** 0.5)
        # attention feature
        feat_reshape = feat.reshape(B, C, W * H)  # B, C, N=WH
        w2v_att = self.w2v_att.to(torch.cuda.current_device())
        query = self.QueryW(w2v_att) # [S, L]*[L,M] -> [S,M]
        query_batch = query.unsqueeze(0).repeat(B, 1, 1)  # [S,M] -> [1,S,M] -> [B,S,M]
        key = self.KeyW(feat_reshape.permute(0, 2, 1))  # [B, C, N] -> [B, N, C] -> [B, N, M]
        value = self.ValueW(feat_reshape.permute(0, 2, 1))  # [B, C, N] -> [B, N, C] -> [B, N, M]

        attention = F.softmax(torch.matmul(query_batch, key.permute(0, 2, 1)),
                              dim=2)  # [B,S,M],[B,N,M] -> [B,S,M],[B,M,N] -> [B, S, N]
        attented_feat = torch.matmul(attention, value)  # [B, S, N] * [B, N, M] -> [B,S,M]
        attented_feat_o = self.W_o(attented_feat)  # [B,S,M] -> [B,S,C]

        feat_pool = F.avg_pool2d(feat, kernel_size=(W, H)).view(B, 1, -1)  # B, C
        feat_reshape_repeat = feat_pool.expand(B, self.attritube_num, C)  # B, S, C
        attented_feat_final = feat_reshape_repeat + self.ratio * attented_feat_o  # [B,S,C]

        # visual to semantic
        if self.hid_dim == 0:
            v2s = torch.einsum('BSC,SC->BS', attented_feat_final, self.V_att_final_branch)  # [B,312,2048] * [312, 2048] -> [B,312,2048] -> [B,312]
        else:
            attented_feat_hid = self.V_att_hidden_branch(attented_feat_final) # [B,312,2048] -> [B,312,4096]
            v2s = torch.einsum('BSH,SH->BS', attented_feat_hid, self.V_att_final_branch)  # [B,312,4096] * [312, 4096] -> [B,312]
        if getAttention:
            return v2s, attention
        else:
            return v2s

    def conv_features(self, x):
        '''
        the feature input to prototype layer
        '''
        if self.backbone_type == 'resnet':
            x = self.backbone(x)  # if resnet, x: [b,w,h,2048], if vit x: [b,w,h,2048]
            return x
        elif self.backbone_type == 'vit':
            x, semantic_feat = self.backbone(x)  # if resnet, x: [b,w,h,2048], if vit x: [b,w,h,2048]
            return x, semantic_feat

    def euclidean_dist(self, prediction, support_att, norm=False):
        if norm == False:
            N, S = prediction.shape
            C, S = support_att.shape

            support_att_expand = support_att.unsqueeze(0).expand(N, C, S)
            prediction_expand = prediction.unsqueeze(1).expand(N, C, S)
            offset = torch.sum((prediction_expand - support_att_expand) ** 2, dim=2)  # [N, C, S]-->[N,C]
            return offset
        else:
            N, S = prediction.shape
            C, S = support_att.shape
            support_att_norm = torch.norm(support_att, p=2, dim=1).unsqueeze(1).expand_as(support_att)
            support_att_normalized = support_att.div(support_att_norm + 1e-10)
            prediction_norm = torch.norm(prediction, p=2, dim=1).unsqueeze(1).expand_as(prediction)
            prediction_normalized = prediction.div(prediction_norm + 1e-10)

            support_att_expand = support_att_normalized.unsqueeze(0).expand(N, C, S)
            prediction_expand = prediction_normalized.unsqueeze(1).expand(N, C, S)
            offset = torch.sum((prediction_expand - support_att_expand) ** 2, dim=2)  # [N, C, S]-->[N,C]

            return offset

    def cosine_dis(self, pred_att, support_att):
        pred_att_norm = torch.norm(pred_att, p=2, dim=1).unsqueeze(1).expand_as(pred_att)
        pred_att_normalized = pred_att.div(pred_att_norm + 1e-10)
        support_att_norm = torch.norm(support_att, p=2, dim=1).unsqueeze(1).expand_as(support_att)
        support_att_normalized = support_att.div(support_att_norm + 1e-10)
        cos_dist = torch.einsum('bd,nd->bn', pred_att_normalized, support_att_normalized)
        score = cos_dist * self.scale  # B, cls_num
        return score, cos_dist


    def check_fine_tune(self):
        if self.ft_flag:
            for p in self.backbone.parameters():
                p.requires_grad = True
        else:
            for p in self.backbone.parameters():
                p.requires_grad = False