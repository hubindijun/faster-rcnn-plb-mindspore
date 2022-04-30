# Copyright 2020-2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""RPN for fasterRCNN"""
import numpy as np
import mindspore as ms
import mindspore.ops as ops
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import numpy as mdnp


from .bbox_assign_sample import BboxAssignSample


class RpnRegClsBlock(nn.Cell):
    """
    Rpn reg cls block for rpn layer

    Args:
        in_channels (int) - Input channels of shared convolution.
        feat_channels (int) - Output channels of shared convolution.
        num_anchors (int) - The anchor number.
        cls_out_channels (int) - Output channels of classification convolution.
        weight_conv (Tensor) - weight init for rpn conv.
        bias_conv (Tensor) - bias init for rpn conv.
        weight_cls (Tensor) - weight init for rpn cls conv.
        bias_cls (Tensor) - bias init for rpn cls conv.
        weight_reg (Tensor) - weight init for rpn reg conv.
        bias_reg (Tensor) - bias init for rpn reg conv.

    Returns:
        Tensor, output tensor.
    """
    def __init__(self,
                 in_channels,
                 feat_channels,
                 num_anchors,
                 cls_out_channels,
                 weight_conv,
                 bias_conv,
                 weight_cls,
                 bias_cls,
                 weight_reg,
                 bias_reg):
        super(RpnRegClsBlock, self).__init__()
        self.rpn_conv = nn.Conv2d(in_channels, feat_channels, kernel_size=3, stride=1, pad_mode='same',
                                  has_bias=True, weight_init=weight_conv, bias_init=bias_conv)
        self.relu = nn.ReLU()

        self.rpn_cls = nn.Conv2d(feat_channels, num_anchors * cls_out_channels, kernel_size=1, pad_mode='valid',
                                 has_bias=True, weight_init=weight_cls, bias_init=bias_cls)
        self.rpn_reg = nn.Conv2d(feat_channels, num_anchors * 4, kernel_size=1, pad_mode='valid',
                                 has_bias=True, weight_init=weight_reg, bias_init=bias_reg)

    def construct(self, x):
        x = self.relu(self.rpn_conv(x))

        x1 = self.rpn_cls(x)
        x2 = self.rpn_reg(x)

        return x1, x2


class RPN(nn.Cell):
    """
    ROI proposal network..

    Args:
        config (dict) - Config.
        batch_size (int) - Batchsize.
        in_channels (int) - Input channels of shared convolution.
        feat_channels (int) - Output channels of shared convolution.
        num_anchors (int) - The anchor number.
        cls_out_channels (int) - Output channels of classification convolution.

    Returns:
        Tuple, tuple of output tensor.

    Examples:
        RPN(config=config, batch_size=2, in_channels=256, feat_channels=1024,
            num_anchors=3, cls_out_channels=512)
    """
    def __init__(self,
                 config,
                 batch_size,
                 in_channels,
                 feat_channels,
                 num_anchors,
                 cls_out_channels):
        super(RPN, self).__init__()
        cfg_rpn = config
        self.config = config
        self.dtype = np.float32
        self.ms_type = ms.float32
        self.device_type = "Ascend" if ms.get_context("device_target") == "Ascend" else "Others"
        self.num_bboxes = cfg_rpn.num_bboxes
        self.slice_index = ()
        self.feature_anchor_shape = ()
        self.slice_index += (0,)
        index = 0
        for shape in cfg_rpn.feature_shapes:
            self.slice_index += (self.slice_index[index] + shape[0] * shape[1] * num_anchors,)
            self.feature_anchor_shape += (shape[0] * shape[1] * num_anchors * batch_size,)
            index += 1

        self.num_anchors = num_anchors
        self.batch_size = batch_size
        self.test_batch_size = cfg_rpn.test_batch_size
        self.num_layers = 5
        self.real_ratio = ms.numpy.ones((1, 1), self.dtype)

        self.rpn_convs_list = nn.layer.CellList(self._make_rpn_layer(self.num_layers, in_channels, feat_channels,
                                                                     num_anchors, cls_out_channels))

        self.transpose = ops.Transpose()
        self.reshape = ops.Reshape()
        self.concat = ops.Concat(axis=0)
        self.fill = ops.Fill()
        self.placeh1 = ms.numpy.ones((1,), self.dtype)

        self.trans_shape = (0, 2, 3, 1)

        self.reshape_shape_reg = (-1, 4)
        self.reshape_shape_cls = (-1,)
        self.rpn_loss_reg_weight = Tensor(np.array(cfg_rpn.rpn_loss_reg_weight).astype(self.dtype))
        self.rpn_loss_cls_weight = Tensor(np.array(cfg_rpn.rpn_loss_cls_weight).astype(self.dtype))
        self.num_expected_total = Tensor(np.array(cfg_rpn.num_expected_neg * self.batch_size).astype(self.dtype))
        self.num_bboxes = cfg_rpn.num_bboxes
        self.get_targets = BboxAssignSample(cfg_rpn, self.batch_size, self.num_bboxes, False)
        self.CheckValid = ops.CheckValid()
        self.sum_loss = ops.ReduceSum()
        self.loss_cls = ops.SigmoidCrossEntropyWithLogits()
        self.loss_bbox = ops.SmoothL1Loss(beta=1.0/9.0)
        self.squeeze = ops.Squeeze()
        self.cast = ops.Cast()
        self.tile = ops.Tile()
        self.zeros_like = ops.ZerosLike()
        self.loss = ms.numpy.zeros((1,), self.dtype)
        self.clsloss = ms.numpy.zeros((1,), self.dtype)
        self.regloss = ms.numpy.zeros((1,), self.dtype)

    def _make_rpn_layer(self, num_layers, in_channels, feat_channels, num_anchors, cls_out_channels):
        """
        make rpn layer for rpn proposal network

        Args:
        num_layers (int) - layer num.
        in_channels (int) - Input channels of shared convolution.
        feat_channels (int) - Output channels of shared convolution.
        num_anchors (int) - The anchor number.
        cls_out_channels (int) - Output channels of classification convolution.

        Returns:
        List, list of RpnRegClsBlock cells.
        """
        rpn_layer = []

        shp_weight_conv = (feat_channels, in_channels, 3, 3)
        shp_bias_conv = (feat_channels,)
        weight_conv = ms.common.initializer.initializer('Normal', shape=shp_weight_conv, dtype=self.ms_type).to_tensor()
        bias_conv = ms.common.initializer.initializer(0, shape=shp_bias_conv, dtype=self.ms_type).to_tensor()

        shp_weight_cls = (num_anchors * cls_out_channels, feat_channels, 1, 1)
        shp_bias_cls = (num_anchors * cls_out_channels,)
        weight_cls = ms.common.initializer.initializer('Normal', shape=shp_weight_cls, dtype=self.ms_type).to_tensor()
        bias_cls = ms.common.initializer.initializer(0, shape=shp_bias_cls, dtype=self.ms_type).to_tensor()

        shp_weight_reg = (num_anchors * 4, feat_channels, 1, 1)
        shp_bias_reg = (num_anchors * 4,)
        weight_reg = ms.common.initializer.initializer('Normal', shape=shp_weight_reg, dtype=self.ms_type).to_tensor()
        bias_reg = ms.common.initializer.initializer(0, shape=shp_bias_reg, dtype=self.ms_type).to_tensor()

        for i in range(num_layers):
            rpn_reg_cls_block = RpnRegClsBlock(in_channels, feat_channels, num_anchors, cls_out_channels, \
                                               weight_conv, bias_conv, weight_cls, \
                                               bias_cls, weight_reg, bias_reg)
            if self.device_type == "Ascend":
                rpn_reg_cls_block.to_float(ms.float16)
            rpn_layer.append(rpn_reg_cls_block)

        for i in range(1, num_layers):
            rpn_layer[i].rpn_conv.weight = rpn_layer[0].rpn_conv.weight
            rpn_layer[i].rpn_cls.weight = rpn_layer[0].rpn_cls.weight
            rpn_layer[i].rpn_reg.weight = rpn_layer[0].rpn_reg.weight

            rpn_layer[i].rpn_conv.bias = rpn_layer[0].rpn_conv.bias
            rpn_layer[i].rpn_cls.bias = rpn_layer[0].rpn_cls.bias
            rpn_layer[i].rpn_reg.bias = rpn_layer[0].rpn_reg.bias

        return rpn_layer

    def construct(self, inputs, img_metas, anchor_list, gt_bboxes, gt_labels, gt_valids):
        #TODO 可以借助img_meta计算坐标位置解码 以及scale一类，参考faster_rcnn lin378
        loss_print = ()
        rpn_cls_score = ()
        rpn_bbox_pred = ()
        rpn_cls_score_total = ()
        rpn_bbox_pred_total = ()
        # Uint8 和Uint64 在cpu内核下，都不支持gt_labels_i 切片， 仅支持float32，
        gt_labels = self.cast(gt_labels, ms.float32)

        for i in range(self.num_layers):
            x1, x2 = self.rpn_convs_list[i](inputs[i])

            rpn_cls_score_total = rpn_cls_score_total + (x1,)
            rpn_bbox_pred_total = rpn_bbox_pred_total + (x2,)

            x1 = self.transpose(x1, self.trans_shape)
            x1 = self.reshape(x1, self.reshape_shape_cls)

            x2 = self.transpose(x2, self.trans_shape)
            x2 = self.reshape(x2, self.reshape_shape_reg)

            rpn_cls_score = rpn_cls_score + (x1,)
            rpn_bbox_pred = rpn_bbox_pred + (x2,)

        loss = self.loss
        clsloss = self.clsloss
        regloss = self.regloss
        bbox_targets = ()
        bbox_weights = ()
        labels = ()
        label_weights = ()
        #存放用以后续解码预测框坐标
        decode_anchor_lists = ()
        output = ()
        if self.training:
            #batch_size = 默认值为2,对应于default_config.yaml的配置项
            for i in range(self.batch_size):
                multi_level_flags = ()
                anchor_list_tuple = ()
                #num_layer是fpn层数-1
                for j in range(self.num_layers):
                    res = self.cast(self.CheckValid(anchor_list[j], self.squeeze(img_metas[i:i + 1:1, ::])),
                                    ms.int32)
                    multi_level_flags = multi_level_flags + (res,)
                    anchor_list_tuple = anchor_list_tuple + (anchor_list[j],)

                valid_flag_list = self.concat(multi_level_flags)
                anchor_using_list = self.concat(anchor_list_tuple)


                gt_bboxes_i = self.squeeze(gt_bboxes[i:i + 1:1, ::])
                gt_labels_i = self.squeeze(gt_labels[i:i + 1:1, ::])
                #切片后再转换回来，还原到uint8
                gt_labels_i = self.cast(gt_labels_i, ms.uint8)
                gt_valids_i = self.squeeze(gt_valids[i:i + 1:1, ::])

                # anchor—_using_list tensor:(245520,4)
                bbox_target, bbox_weight, label, label_weight = self.get_targets(gt_bboxes_i,
                                                                                 gt_labels_i,
                                                                                 self.cast(valid_flag_list,
                                                                                           ms.bool_),
                                                                                 anchor_using_list, gt_valids_i)
               # anchor_using_list 和 bbox_target 对应不上,应该通过for循环中，取出最早的list中的两个，再contact出来2倍的
               # bbox_target是追加了部分真值标签的
                bbox_target = self.cast(bbox_target, self.ms_type)
                bbox_weight = self.cast(bbox_weight, self.ms_type)
                label = self.cast(label, self.ms_type)
                label_weight = self.cast(label_weight, self.ms_type)

                for j in range(self.num_layers):
                    begin = self.slice_index[j]
                    end = self.slice_index[j + 1]
                    stride = 1
                    bbox_targets += (bbox_target[begin:end:stride, ::],)
                    bbox_weights += (bbox_weight[begin:end:stride],)
                    labels += (label[begin:end:stride],)
                    label_weights += (label_weight[begin:end:stride],)
                    decode_anchor_lists +=(anchor_using_list[begin:end:stride, ::],)

            #防止出现anchor都是0的结果导致nan错误，待研究
            for i in range(self.num_layers):
                bbox_target_using = ()
                bbox_weight_using = ()
                label_using = ()
                label_weight_using = ()
                temp_anchor_list = ()

                for j in range(self.batch_size):
                    bbox_target_using += (bbox_targets[i + (self.num_layers * j)],)
                    bbox_weight_using += (bbox_weights[i + (self.num_layers * j)],)
                    label_using += (labels[i + (self.num_layers * j)],)
                    label_weight_using += (label_weights[i + (self.num_layers * j)],)
                    temp_anchor_list += (decode_anchor_lists[i + (self.num_layers * j)],)

                bbox_target_with_batchsize = self.concat(bbox_target_using)
                bbox_weight_with_batchsize = self.concat(bbox_weight_using)
                label_with_batchsize = self.concat(label_using)
                label_weight_with_batchsize = self.concat(label_weight_using)
                temp_anchor_list_with_batchsize = self.concat(temp_anchor_list)
                # stop
                bbox_target_ = ops.stop_gradient(bbox_target_with_batchsize)
                bbox_weight_ = ops.stop_gradient(bbox_weight_with_batchsize)
                label_ = ops.stop_gradient(label_with_batchsize)
                label_weight_ = ops.stop_gradient(label_weight_with_batchsize)
                decode_anchor_list_ = ops.stop_gradient(temp_anchor_list_with_batchsize)

                cls_score_i = self.cast(rpn_cls_score[i], self.ms_type)
                reg_score_i = self.cast(rpn_bbox_pred[i], self.ms_type)

                loss_cls = self.loss_cls(cls_score_i, label_)
                loss_cls_item = loss_cls * label_weight_

                #获取预测框坐标，计算PLB系数
                #理想算法实现1： 使用基于框架底层c++实现的api，使用解码ops方法
                # 缺陷：由于mindspore中的BoundingBoxDecode暂不支持反向传播属性，导致静态图模式下训练时出现错误，而调试模式可执行。
                # bounding_box_decode = ops.BoundingBoxDecode(max_shape=(self.config.img_height, self.config.img_width), means=tuple(self.config.rpn_target_means),
                #                           stds=tuple(self.config.rpn_target_stds))
                # predicted_boxes_decode = bounding_box_decode(decode_anchor_list_,reg_score_i)
                # predicted_boxes = ops.stop_gradient(predicted_boxes_decode)
                # box_width = predicted_boxes[:, 2] - predicted_boxes[:, 0]
                # box_height = predicted_boxes[:, 3] - predicted_boxes[:, 1]

                #手动计算方法2:已知anchor中的wh和pred_deltas偏移dw,dh,求面积结果
                #缺陷：由于Mindspore-GPU环境的图模式运算下，不支持np.asnumpy()方法，同时没有合适的指数运算exp方法，
                #备注：CPU环境中正常
                #手写计算方法2.通过anchor和pred_offset计算出面积
                # box_width_anchor = decode_anchor_list_[:, 2]
                # box_height_anchor = decode_anchor_list_[:, 3]
                # #由于mindspore中的Tensor没有指数函数操作，因此借助与numpy转换运算
                # w_scale = np.exp(reg_score_i[:, 2].asnumpy())
                # w_scale = Tensor(w_scale,dtype=self.ms_type)
                # h_scale = np.exp(reg_score_i[:, 3].asnumpy())
                # h_scale = Tensor(h_scale,dtype=self.ms_type)
                # #计算检测框的宽度和高度
                # box_width = box_width_anchor * w_scale
                # box_height = box_height_anchor * h_scale

                #手写计算方法3：直接用anchor的大小来作为预测框大小，未来可以优化为前两种方法
                box_width_anchor = decode_anchor_list_[:, 2]
                box_height_anchor = decode_anchor_list_[:, 3]
                # 计算检测框的宽度和高度
                box_width = box_width_anchor
                box_height = box_height_anchor

                # 得到对应的检测框面积矩阵
                predicted_boxes_aeras = box_width * box_height
                #直接对面积矩阵中，对应label_weight_为0的部分，设置面积为0，其余部分统计相对的平均值
                #不能直接*label_weight，因为可能导致所有的areas都变成0，造成后续PLB计算时，负无穷溢出
                # 统计出对应label_weight_中为1的索引列的值，求平均值，label_weight_=0的部分不应干扰平均值的计算
                mean_area = predicted_boxes_aeras.mean()
                #平均面积的求解，可以优化为，获取label_weight_中对应序号的，不为0的检测框，求其和，并除以对应的数目
                #进一步排除其他不参与损失计算的候选框的干扰
                # 此处求mean时对于个别层预测框几乎都为0的情况，做补偿操作，防止除0导致正负无穷的结果，训练产生Nan错误
                if mean_area < 0.1:
                    print("every box is 0,set PLB weight as label_weight_ .avoid divide zero error.")
                    PLB_weight_i = label_weight_
                else :
                    PLB_weight_i = (mean_area*2) / (mean_area+predicted_boxes_aeras)

                # 对比未经过PLB的损失结果#
                loss_cls_item_before_plb =  ops.stop_gradient(loss_cls_item)
                loss_cls_item_before_plb =  self.sum_loss(loss_cls_item_before_plb, (0,)) / self.num_expected_total
                print(loss_cls_item_before_plb)

                #对于mindspore-gpu下，tensor和nparray的转换，以及mindspore.numpy的区别
                loss_cls_item = loss_cls_item * PLB_weight_i
                loss_cls_item = self.sum_loss(loss_cls_item, (0,)) / self.num_expected_total

                loss_reg = self.loss_bbox(reg_score_i, bbox_target_)
                bbox_weight_ = self.tile(self.reshape(bbox_weight_, (self.feature_anchor_shape[i], 1)), (1, 4))
                #loss_reg 是个n*4的tensor
                loss_reg = loss_reg * bbox_weight_
                loss_reg_item = self.sum_loss(loss_reg, (1,))
                #RPN边框回归损失函数加入PLB操作
                loss_reg_item = loss_reg_item * PLB_weight_i
                loss_reg_item = self.sum_loss(loss_reg_item, (0,)) / self.num_expected_total

                loss_total = self.rpn_loss_cls_weight * loss_cls_item + self.rpn_loss_reg_weight * loss_reg_item

                loss += loss_total
                loss_print += (loss_total, loss_cls_item, loss_reg_item)
                clsloss += loss_cls_item
                regloss += loss_reg_item

                print(loss_cls_item)
                print(clsloss)
                print('-----')
                output = (loss, rpn_cls_score_total, rpn_bbox_pred_total, clsloss, regloss, loss_print)
        else:
            output = (self.placeh1, rpn_cls_score_total, rpn_bbox_pred_total, self.placeh1, self.placeh1, self.placeh1)

        return output


