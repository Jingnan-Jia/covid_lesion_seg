# -*- coding: utf-8 -*-
"""
Compiled models for different tasks.
=============================================================
Created on Tue Apr  4 09:35:14 2017
@author: Jingnan
"""

from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Activation, Lambda, Dropout, Conv3D, BatchNormalization, add, concatenate, \
    UpSampling3D, PReLU
from tensorflow import multiply
from tensorflow.keras import backend as K
import tensorflow as tf
import os
from tensorflow.keras.losses import categorical_crossentropy

# I do not know if this line should be put here or in the train.py, so I put it both
# tf.keras.mixed_precision.experimental.set_policy('infer')
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'


# Ref: salehi17, "Twersky loss function for image segmentation using 3D FCDN"
# -> the score is computed for each class separately and then summed
# alpha=beta=0.5 : dice coefficient
# alpha=beta=1   : tanimoto coefficient (also known as jaccard)
# alpha+beta=1   : produces set of F*-scores
# implemented by E. Moebel, 06/04/18
def dice_coef_every_class(y_true,
                          y_pred):  # not used yet, because one metric must be one scalar from what I experienced
    """
    dice coefficient for every class/label.

    :param y_true: ground truth
    :param y_pred: prediction results
    :return: dice value
    """
    alpha = 0.5
    beta = 0.5

    ones = K.ones(K.shape(y_true))
    p0 = y_pred  # proba that voxels are class i
    p1 = ones - y_pred  # proba that voxels are not class i
    g0 = y_true
    g1 = ones - y_true

    num = K.sum(p0 * g0, (0, 1, 2, 3))
    den = num + alpha * K.sum(p0 * g1, (0, 1, 2, 3)) + beta * K.sum(p1 * g0, (0, 1, 2, 3))

    T = K.sum(num / den)  # when summing over classes, T has dynamic range [0 Ncl]

    Ncl = K.cast(K.shape(y_true)[-1], 'float32')
    return Ncl - T


def dice_coef(y_true, y_pred):  # Not used, because it include background, meaningless
    """

    Returns overall dice coefficient after supressing the background

    TODO : Choose channel(and axis) of background

    :param y_true:
    :param y_pred:
    :return:
    """
    y_true_f = K.flatten(Lambda(lambda y_true: y_true[:, :, :, :, 0:])(y_true))
    y_pred_f = K.flatten(Lambda(lambda y_pred: y_pred[:, :, :, :, 0:])(y_pred))

    smooth = 0.0001
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_wo_bg(y_true, y_pred):  # not used, it is sum, but we need average
    """
    Returns overall dice coefficient after supressing the background

    TODO : Choose channel(and axis) of background
    """
    y_true_f = K.flatten(Lambda(lambda y_true: y_true[:, :, :, :, 1:])(y_true))
    y_pred_f = K.flatten(Lambda(lambda y_pred: y_pred[:, :, :, :, 1:])(y_pred))

    smooth = 0.0001
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_prod(y_true, y_pred):
    """
    completely same with dice_coef but cahieved with different operations

    TODO : Choose channel (and axis) of background
           Choose other merge methods (sum,avg,etc)
    """
    y_true_f = (Lambda(lambda y_true: y_true[:, :, :, :, 0:])(y_true))
    y_pred_f = (Lambda(lambda y_pred: y_pred[:, :, :, :, 0:])(y_pred))

    product = multiply([y_true_f, y_pred_f])

    red_y_true = K.sum(y_true_f, axis=[0, 1, 2, 3])
    red_y_pred = K.sum(y_pred_f, axis=[0, 1, 2, 3])
    red_product = K.sum(product, axis=[0, 1, 2, 3])

    smooth = 0.0001
    dices = (2. * red_product + smooth) / (red_y_true + red_y_pred + smooth)

    return K.prod(dices)


def dice_coef_every_class_old(y_true, y_pred):  # not used
    """
    Returns overall dice coefficient after supressing the background

    TODO : Choose channel(and axis) of background
    """
    y_true_f = (Lambda(lambda y_true: y_true[:, :, :, :, 0:])(y_true))
    y_pred_f = (Lambda(lambda y_pred: y_pred[:, :, :, :, 0:])(y_pred))

    product = multiply([y_true_f, y_pred_f])

    red_y_true = K.sum(y_true_f, axis=[0, 1, 2, 3])
    red_y_pred = K.sum(y_pred_f, axis=[0, 1, 2, 3])
    red_product = K.sum(product, axis=[0, 1, 2, 3])

    smooth = 0.0001
    dices = (2. * red_product + smooth) / (red_y_true + red_y_pred + smooth)

    return dices


def dice_coef_weight_sub(y_true, y_pred):  # what is the difference between tf.multiply and .*??
    """
    Returns the product of dice coefficient for each class
    this dice is designed to increase the ratio of class which include smaller area,
    by the function: ratio_y_pred = 1.0 - ratio_y_pred)
    hope this can make the model pay more attention to the small classes like the right middle lobe
    can be used but not powerful as power weights

    TODO : Choose channel (and axis) of background
           Choose other merge methods (sum,avg,etc)
    """
    y_true_f = (Lambda(lambda y_true: y_true[:, :, :, :, 0:])(y_true))
    y_pred_f = (Lambda(lambda y_pred: y_pred[:, :, :, :, 0:])(y_pred))

    product = multiply([y_true_f, y_pred_f])  # multiply should be import from tf or tf.math

    red_y_true = K.sum(y_true_f, axis=[0, 1, 2, 3])  # shape [None, nb_class]
    red_y_pred = K.sum(y_pred_f, axis=[0, 1, 2, 3])
    red_product = K.sum(product, axis=[0, 1, 2, 3])

    smooth = 0.0001
    dices = (2. * red_product + smooth) / (red_y_true + red_y_pred + smooth)

    ratio = red_y_true / (K.sum(red_y_true) + smooth)
    ratio_y_pred = 1.0 - ratio
    ratio_y_pred = ratio_y_pred / K.sum(ratio_y_pred)  # I do not understand it, K.sum(ratio_y_pred) = 1?

    return K.sum(multiply([dices, ratio_y_pred]))


def dice_coef_weight_pow(y_true, y_pred):
    """
    this dice is designed to increase the ratio of class which include smaller area,
    by the function: K.pow(ratio_y_pred + 0.001, -1.0)
    hope this can make the model pay more attention to the small classes like the right middle lobe

    TODO : Choose channel (and axis) of background
           Choose other merge methods (sum,avg,etc)
    """
    y_true_f = (Lambda(lambda y_true: y_true[:, :, :, :, 0:])(y_true))
    y_pred_f = (Lambda(lambda y_pred: y_pred[:, :, :, :, 0:])(y_pred))

    product = multiply(y_true_f, y_pred_f)

    red_y_true = K.sum(y_true_f, axis=[0, 1, 2, 3])
    red_y_pred = K.sum(y_pred_f, axis=[0, 1, 2, 3])
    red_product = K.sum(product, axis=[0, 1, 2, 3])

    smooth = 0.0001
    dices = (2. * red_product + smooth) / (red_y_true + red_y_pred + smooth)

    ratio_y_pred = red_y_true / (K.sum(red_y_true) + smooth)
    ratio_y_pred = K.pow(ratio_y_pred + 0.001, -1.0)  # Here can I use 1.0/(ratio + 0.001)?
    ratio_y_pred = ratio_y_pred / K.sum(ratio_y_pred)

    return K.sum(multiply(dices, ratio_y_pred))


def dices_all_class(y_true, y_pred):
    '''

    :param y_true:
    :param y_pred:
    :return: dices list
    '''
    y_true_f = (Lambda(lambda y_true: y_true[:, :, :, :, 0:])(y_true))
    y_pred_f = (Lambda(lambda y_pred: y_pred[:, :, :, :, 0:])(y_pred))

    product = multiply(y_true_f, y_pred_f)

    red_y_true = K.sum(y_true_f, axis=[0, 1, 2, 3])
    red_y_pred = K.sum(y_pred_f, axis=[0, 1, 2, 3])
    red_product = K.sum(product, axis=[0, 1, 2, 3])

    smooth = 0.0001
    dices = (2. * red_product + smooth) / (red_y_true + red_y_pred + smooth)

    return dices


def dice_0(y_true, y_pred):
    dices = dices_all_class(y_true, y_pred)
    return dices[0]  # return the average dice over the total classes


def dice_1(y_true, y_pred):
    dices = dices_all_class(y_true, y_pred)
    return dices[1]  # return the average dice over the total classes


def dice_2(y_true, y_pred):
    dices = dices_all_class(y_true, y_pred)
    return dices[2]  # return the average dice over the total classes


def dice_3(y_true, y_pred):
    dices = dices_all_class(y_true, y_pred)
    return dices[3]  # return the average dice over the total classes


def dice_4(y_true, y_pred):
    dices = dices_all_class(y_true, y_pred)
    return dices[4]  # return the average dice over the total classes


def dice_5(y_true, y_pred):
    dices = dices_all_class(y_true, y_pred)
    return dices[5]  # return the average dice over the total classes


def dice_coef_mean(y_true, y_pred):
    """
    Returns the product of dice coefficient for each class
    it assumes channel 0 as background

    TODO : Choose channel (and axis) of background
           Choose other merge methods (sum,avg,etc)
    """

    y_true_f = (Lambda(lambda y_true: y_true[:, :, :, :, 1:])(y_true))
    y_pred_f = (Lambda(lambda y_pred: y_pred[:, :, :, :, 1:])(y_pred))

    product = multiply(y_true_f, y_pred_f)

    red_y_true = K.sum(y_true_f, axis=[0, 1, 2, 3])
    red_y_pred = K.sum(y_pred_f, axis=[0, 1, 2, 3])
    red_product = K.sum(product, axis=[0, 1, 2, 3])

    smooth = 0.0001
    dices = (2. * red_product + smooth) / (red_y_true + red_y_pred + smooth)
    mean = K.sum(dices)

    return K.mean(dices)  # return the average dice over the total classes


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def dice_coef_loss_weight_p(y_true, y_pred):
    return 1 - dice_coef_weight_pow(y_true, y_pred) + categorical_crossentropy(y_true, y_pred)


def intro(inputs, nf, bn, name='1'):
    """
    introduction convolution + PReLU (+ batch_normalization).

    :param inputs: data from last output
    :param nf: number of filters (or feature maps)
    :param bn: if batch normalization
    :return: convolution results
    """
    # inputs = Input((None, None, None, nch))
    conv = Conv3D(nf, 3, padding='same', name='intro_Conv3D' + name)(inputs)
    conv = PReLU(shared_axes=[1, 2, 3], name='intro_PReLU' + name)(conv)
    if bn:
        conv = BatchNormalization(name='intro_bn' + name)(conv)
    # m = Model(inputs=inputs, outputs=c
    return conv


def down_trans(inputs, nf, nconvs, bn, dr, ty='v', name='block'):
    # inputs = Input((None, None, None, nch))

    downconv = Conv3D(nf, 2, padding='valid', strides=(2, 2, 2), name=name + '_Conv3D_0')(inputs)
    downconv = PReLU(shared_axes=[1, 2, 3], name=name + '_PReLU_0')(downconv)
    if bn:
        downconv = BatchNormalization(name=name + '_bn_0')(downconv)
    if dr:
        downconv = Dropout(0.5, name=name + '_dr_0')(downconv)

    conv = downconv
    for i in range(nconvs):
        conv = Conv3D(nf, 3, padding='same', name=name + '_Conv3D_' + str(i + 1))(conv)
        conv = PReLU(shared_axes=[1, 2, 3], name=name + '_PReLU_' + str(i + 1))(conv)
        if bn:
            conv = BatchNormalization(name=name + '_bn_' + str(i + 1))(conv)

    if ty == 'v':  # V-Net
        d = add([conv, downconv])
    elif ty == 'u':  # U-Net
        d = conv
    else:
        raise Exception("please assign the model net_type: 'v' or 'u'.")

    # m = Model(inputs=inputs, outputs=d)

    return d


def up_trans(input1, nf, nconvs, bn, dr, ty='v', input2=None, name='block'):
    """
    up transform convolution.

    :param input1: output from last layer
    :param nf: number of filters (or feature maps)
    :param nconvs: number of convolution operations in each level apart from basic downconvolutin or upconvolution.
    :param bn: if batch normalization
    :param dr: if dropout
    :param ty: model type, 'v' means V-Net, 'u' means 'U-Net'
    :param input2: second input, necessary for merge results from short connection
    :param name: block name
    :return: convolution results
    """

    upconv = UpSampling3D((2, 2, 2), name=name + '_Upsampling3D')(input1)  #
    upconv = Conv3D(nf, 2, padding='same', name=name + '_Conv3D_0')(upconv)
    upconv = PReLU(shared_axes=[1, 2, 3], name=name + '_PReLU_0')(upconv)
    if bn:
        upconv = BatchNormalization(name=name + '_bn_0')(upconv)
    if dr:
        upconv = Dropout(0.5, name=name + '_dr_0')(upconv)

    if input2 is not None:
        conv = concatenate([upconv, input2])
    else:
        conv = upconv
    for i in range(nconvs):
        conv = Conv3D(nf, 3, padding='same', name=name + '_Conv3D_' + str(i + 1))(conv)
        conv = PReLU(shared_axes=[1, 2, 3], name=name + '_PReLU_' + str(i + 1))(conv)
        if bn:
            conv = BatchNormalization(name=name + '_bn_' + str(i + 1))(conv)

    if ty == 'v':  # V-Net
        d = add([conv, upconv])
    elif ty == 'u':  # U-Net
        d = conv
    else:
        raise Exception("please assign the model net_type: 'v' or 'u'.")
    return d


def get_loss_weights_optim(ao, ds, lr, io, task=None):
    if task is "recon":
        loss = ["MSE"]
    else:
        loss = [dice_coef_loss_weight_p]
    loss_weights = [1]
    if "2_out" in io:
        if task == "recon":
            loss = ['mse']
        else:
            loss.append(dice_coef_loss_weight_p)
        loss_weights = [0.5, 0.5]
    if ao:
        loss.append(dice_coef_loss_weight_p)
        loss_weights = [0.5, 0.5]
        if ds == 2:
            loss.append(dice_coef_loss_weight_p)
            loss.append(dice_coef_loss_weight_p)
            loss_weights = [0.375, 0.375, 0.125, 0.125]
    elif ds == 2:
        loss.append(dice_coef_loss_weight_p)
        loss.append(dice_coef_loss_weight_p)
        loss_weights = [0.5, 0.25, 0.25]

    loss_itgt_recon = loss + ['mse']
    loss_itgt_recon_weights = loss_weights + [0.1 * loss_weights[0]]

    optim_tmp = Adam(lr)
    optim = tf.train.experimental.enable_mixed_precision_graph_rewrite(optim_tmp)

    return loss, loss_weights, loss_itgt_recon, loss_itgt_recon_weights, optim


def up_series(task, dwn_tr4, dwn_tr3, dwn_tr2, dwn_tr1, in_tr, nf, bn, dr, net_type):
    up_tr4 = up_trans(dwn_tr4, nf * 8, 2, bn, dr, ty=net_type, input2=dwn_tr3, name=task + '_block5')
    up_tr3 = up_trans(up_tr4, nf * 4, 2, bn, dr, ty=net_type, input2=dwn_tr2, name=task + '_block6')
    up_tr2 = up_trans(up_tr3, nf * 2, 2, bn, dr, ty=net_type, input2=dwn_tr1, name=task + '_block7')
    up_tr1 = up_trans(up_tr2, nf * 1, 1, bn, dr, ty=net_type, input2=in_tr, name=task + '_block8')
    return up_tr4, up_tr3, up_tr2, up_tr1


def decoder(task, dwn_tr4, dwn_tr3, dwn_tr2, dwn_tr1, in_tr, nf, bn, dr, net_type,
            io, ao, ds, chn):
    # decoder for segmentation. up_path for segmentation V-Net, with shot connections
    up_tr4, up_tr3, up_tr2, up_tr1 = up_series(task, dwn_tr4, dwn_tr3, dwn_tr2, dwn_tr1, in_tr, nf, bn, dr, net_type)

    # classification
    if task is "recon":
        out = Conv3D(chn, 1, padding='same', name=task + '_out_recon')(up_tr1)
    else:
        res = Conv3D(chn, 1, padding='same', name=task + '_Conv3D_last')(up_tr1)
        out = Activation('softmax', name=task + '_out_segmentation')(res)

    out = [out]  # convert to list to append other outputs
    if "2_out" in io:
        res_2 = Conv3D(chn, 1, padding='same', name=task + '_Conv3D_last2')(up_tr1)
        out_2 = Activation('softmax', name=task + '_out_segmentation2')(res_2)
        out.append(out_2)
    if ao:
        # aux_output
        aux_res = Conv3D(2, 1, padding='same', name=task + '_aux_Conv3D_last')(up_tr1)
        aux_out = Activation('softmax', name=task + '_aux')(aux_res)
        out.append(aux_out)
    if ds:
        out = [out]
        # deep supervision#1
        deep_1 = UpSampling3D((2, 2, 2), name=task + '_d1_UpSampling3D_0')(up_tr2)
        res = Conv3D(chn, 1, padding='same', name=task + '_d1_Conv3D_last')(deep_1)
        d_out_1 = Activation('softmax', name=task + '_d1')(res)
        out.append(d_out_1)

        # deep supervision#2
        deep_2 = UpSampling3D((2, 2, 2), name=task + '_d2_UpSampling3D_0')(up_tr3)
        deep_2 = UpSampling3D((2, 2, 2), name=task + '_d2_UpSampling3D_1')(deep_2)
        res = Conv3D(chn, 1, padding='same', name=task + '_d2_Conv3D_last')(deep_2)
        d_out_2 = Activation('softmax', name=task + '_d2')(res)
        out.append(d_out_2)

    return out


def compile_model(task, in_conv, out_conv, out_itgt_conv, io, ao, ds, chn, lr):
    if chn == 6:
        metrics = [dice_coef_mean, dice_0, dice_1, dice_2, dice_3, dice_4, dice_5]
    elif chn == 2:
        metrics = [dice_coef_mean, dice_0, dice_1]
    else:
        metrics = 'mse'
    if task is "recon":
        metrics_dict = {task + "_out_recon": metrics}
    else:
        metrics_dict = {task + '_out_segmentation': metrics}
    if "2_out" in io:
        metrics_dict[task + '_out_segmentation2'] = metrics
    if ao:
        metrics_dict[task + '_aux'] = metrics
    if ds == 2:
        metrics_dict[task + '_d1'] = metrics
        metrics_dict[task + '_d2'] = metrics

    loss, loss_weights, loss_itgt, loss_itgt_weights, optim = get_loss_weights_optim(ao, ds, lr, io)
    net = Model(in_conv, out_conv, name=task)
    net.compile(optimizer=optim,
                loss=loss,
                loss_weights=loss_weights,
                metrics=metrics_dict)

    if task is not "recon":
        net_itgt = Model(in_conv, out_itgt_conv, name=task + "_itgt")
        net_itgt.compile(optimizer=optim,
                         loss=loss_itgt,
                         loss_weights=loss_itgt_weights,
                         metrics=metrics_dict)
    
        return net, net_itgt
    else:
        return net


def load_cp_models(model_names, args):
    """
    load compiled models.
    """

    nch = 1
    nf = args.feature_number
    bn = args.batch_norm
    dr = args.dropout
    net_type = args.u_v
    attention = args.attention

    ## start model
    input_data = Input((None, None, None, nch), name='input')  # input data
    in_tr = intro(input_data, nf, bn)

    if "2_in" in args.lb_io:
        input_data2 = Input((None, None, None, nch), name='input_2')  # input data 2 with a different scale
        in_tr2 = intro(input_data2, nf, bn, name='2')

        in_tr = concatenate([in_tr, in_tr2])
        input_data = [input_data, input_data2]

    # down_path
    dwn_tr1 = down_trans(in_tr, nf * 2, 2, bn, dr, ty=net_type, name='block1')
    dwn_tr2 = down_trans(dwn_tr1, nf * 4, 2, bn, dr, ty=net_type, name='block2')
    dwn_tr3 = down_trans(dwn_tr2, nf * 8, 2, bn, dr, ty=net_type, name='block3')
    dwn_tr4 = down_trans(dwn_tr3, nf * 16, 2, bn, dr, ty=net_type, name='block4')

    #######################################################-----------------------#####################################
    out_lesion = decoder("lesion", dwn_tr4, dwn_tr3, dwn_tr2, dwn_tr1, in_tr, nf, bn, dr, net_type,
                         io=args.ls_io, ao=args.ao_ls, ds=args.ds_ls, chn=2)
    out_lobe = decoder("lobe", dwn_tr4, dwn_tr3, dwn_tr2, dwn_tr1, in_tr, nf, bn, dr, net_type,
                       io=args.lb_io, ao=args.ao_lb, ds=args.ds_lb, chn=6)
    out_lung = decoder("lung", dwn_tr4, dwn_tr3, dwn_tr2, dwn_tr1, in_tr, nf, bn, dr, net_type,
                       io=args.lu_io, ao=args.ao_lu, ds=args.ds_lu, chn=2)
    out_vessel = decoder("vessel", dwn_tr4, dwn_tr3, dwn_tr2, dwn_tr1, in_tr, nf, bn, dr, net_type,
                         io=args.vs_io, ao=args.ao_vs, ds=args.ds_vs, chn=2)
    out_airway = decoder("airway", dwn_tr4, dwn_tr3, dwn_tr2, dwn_tr1, in_tr, nf, bn, dr, net_type,
                         io=args.aw_io, ao=args.ao_aw, ds=args.ds_aw, chn=2)
    out_recon = decoder("recon", dwn_tr4, None, None, None, None, nf, bn, dr, net_type,
                        io=args.rc_io, ao=args.ao_rc, ds=args.ds_rc, chn=1)

    # out_rec = Activation ('softmax', name='rec_out_segmentation') (res_rec) # no activation for reconstruction
    out_itgt_lesion_recon = out_lesion + [out_recon]
    out_itgt_lobe_recon = out_lobe + [out_recon]
    out_itgt_vessel_recon = out_vessel + [out_recon]
    out_itgt_airway_recon = out_airway + [out_recon]
    out_itgt_lung_recon = out_lung + [out_recon]

    net_lesion, net_itgt_lesion_recon = compile_model("lesion", input_data, out_lesion, out_itgt_lesion_recon,
                                                           args.ls_io, args.ao_ls, args.ds_ls, 2, args.lr_ls)
    net_lobe, net_itgt_lobe_recon = compile_model("lobe", input_data, out_lobe, out_itgt_lobe_recon,
                                                       args.lb_io, args.ao_lb, args.ds_lb, 6, args.lr_lb)
    net_vessel, net_itgt_vessel_recon = compile_model("vessel", input_data, out_vessel, out_itgt_vessel_recon,
                                                           args.vs_io, args.ao_vs, args.ds_vs, 2, args.lr_vs)
    net_airway, net_itgt_airway_recon = compile_model("airway", input_data, out_airway, out_itgt_airway_recon,
                                                           args.aw_io, args.ao_aw, args.ds_aw, 2, args.lr_aw)
    net_lung, net_itgt_lung_recon = compile_model("lung", input_data, out_lung, out_itgt_lung_recon,
                                                       args.lu_io, args.ao_lu, args.ds_lu, 2, args.lr_lu)
    net_recon = compile_model("recon", input_data, out_recon, None, args.rc_io, args.ao_rc, args.ds_rc, 1, args.lr_rc)

    models_dict = {
        "net_itgt_ls_rc": net_itgt_lesion_recon,
        "net_itgt_lu_rc": net_itgt_lung_recon,
        "net_itgt_aw_rc": net_itgt_airway_recon,
        "net_itgt_lb_rc": net_itgt_lobe_recon,
        "net_itgt_vs_rc": net_itgt_vessel_recon,

        "net_recon": net_recon,

        "net_lesion": net_lesion,
        "net_lobe": net_lobe,
        "net_vessel": net_vessel,
        "net_lung": net_lung,
        "net_airway": net_airway,
    }

    return list(map(models_dict.get, model_names))


def main():
    print('this is main function')


if __name__ == '__main__':
    main()
