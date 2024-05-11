import torch.nn as nn

from detectron2.config import LazyCall as L
from detectron2.layers import ShapeSpec
from detectron2.modeling.backbone import BasicStem, ResNet
from detrex.layers import PositionEmbeddingSine
from detrex.modeling.matcher import HungarianMatcher
from detrex.modeling.neck import ChannelMapper
from ape.modeling.deta import (
    DeformableCriterion,
    DeformableDETR,
    DeformableDETRSegm,
    DeformableDetrTransformer,
    DeformableDetrTransformerDecoder,
    DeformableDetrTransformerEncoder,
    Stage1Assigner,
    Stage2Assigner,
)

model = L(DeformableDETRSegm)(
    backbone=L(ResNet)(
        stem=L(BasicStem)(in_channels=3, out_channels=64, norm="FrozenBN"),
        stages=L(ResNet.make_default_stages)(
            depth=50,
            stride_in_1x1=False,
            norm="FrozenBN",
        ),
        out_features=["res2", "res3", "res4", "res5"],
        freeze_at=1,
    ),
    position_embedding=L(PositionEmbeddingSine)(
        num_pos_feats=128,
        temperature=10000,
        normalize=True,
        offset=-0.5,
    ),
    neck=L(ChannelMapper)(
        input_shapes={
            "res3": ShapeSpec(channels=512),
            "res4": ShapeSpec(channels=1024),
            "res5": ShapeSpec(channels=2048),
        },
        in_features=["res3", "res4", "res5"],
        out_channels=256,
        num_outs=5,
        kernel_size=1,
        norm_layer=L(nn.GroupNorm)(num_groups=32, num_channels=256),
    ),
    transformer=L(DeformableDetrTransformer)(
        encoder=L(DeformableDetrTransformerEncoder)(
            embed_dim=256,
            num_heads=8,
            feedforward_dim=2048,
            attn_dropout=0.0,
            ffn_dropout=0.0,
            num_layers=6,
            post_norm=False,
            num_feature_levels="${..num_feature_levels}",
        ),
        decoder=L(DeformableDetrTransformerDecoder)(
            embed_dim=256,
            num_heads=8,
            feedforward_dim=2048,
            attn_dropout=0.0,
            ffn_dropout=0.0,
            num_layers=6,
            return_intermediate=True,
            num_feature_levels="${..num_feature_levels}",
        ),
        as_two_stage="${..as_two_stage}",
        num_feature_levels=5,
        two_stage_num_proposals="${..num_queries}",
        assign_first_stage=True,
    ),
    embed_dim=256,
    num_classes=80,
    num_queries=900,
    aux_loss=True,
    with_box_refine=True,
    as_two_stage=True,
    criterion=L(DeformableCriterion)(
        num_classes=80,
        matcher=L(HungarianMatcher)(
            cost_class=2.0,
            cost_bbox=5.0,
            cost_giou=2.0,
            cost_class_type="focal_loss_cost",
            alpha=0.25,
            gamma=2.0,
        ),
        matcher_stage1=L(Stage1Assigner)(
            t_low=0.3,
            t_high=0.7,
            max_k=4,
        ),
        matcher_stage2=L(Stage2Assigner)(
            num_queries="${...num_queries}",
            num_classes="${...num_classes}",
            max_k=4,
        ),
        weight_dict={
            "loss_class": 1.0,
            "loss_bbox": 5.0,
            "loss_giou": 2.0,
            "loss_mask": 5,
            "loss_dice": 5,
        },
        loss_class_type="focal_loss",
        alpha=0.25,
        gamma=2.0,
        losses=["class", "boxes", "masks"],
    ),
    pixel_mean=[123.675, 116.280, 103.530],
    pixel_std=[58.395, 57.120, 57.375],
    select_box_nums_for_evaluation=100,
    input_format="RGB",
    mask_encode_level=0,
    segm_type="maskdino",
    mask_in_features=["res2"],
    input_shapes={
        "res2": ShapeSpec(channels=256),
        "res3": ShapeSpec(channels=512),
        "res4": ShapeSpec(channels=1024),
        "res5": ShapeSpec(channels=2048),
    },
    output_dir=None,
    vis_period=12800,
)

mask_dino_loss = False

mask_dino_single_loss = False

mask_combine_loss = False

if mask_dino_loss:
    model.criterion.weight_dict = {
        "loss_class": 4.0,
        "loss_bbox": 5.0,
        "loss_giou": 2.0,
        "loss_mask_maskdino": 5,
        "loss_dice_maskdino": 5,
    }
    model.criterion.losses = ["class", "boxes", "masks_maskdino"]

if mask_dino_single_loss:
    model.criterion.weight_dict = {
        "loss_class": 1.0,
        "loss_bbox": 5.0,
        "loss_giou": 2.0,
        "loss_mask_maskdino": 5,
        "loss_dice_maskdino": 5,
    }
    model.criterion.losses = ["class", "boxes", "masks_maskdino"]

if mask_combine_loss:
    model.criterion.weight_dict = {
        "loss_class": 1.0,
        "loss_bbox": 5.0,
        "loss_giou": 2.0,
        "loss_mask": 5.0,
        "loss_dice": 5.0,
        "loss_mask_maskdino": 1,
        "loss_dice_maskdino": 1,
    }
    model.criterion.losses = ["class", "boxes", "masks", "masks_maskdino"]

if model.aux_loss:
    weight_dict = model.criterion.weight_dict
    aux_weight_dict = {}
    for i in range(model.transformer.decoder.num_layers - 1):
        aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
    aux_weight_dict.update({k + "_enc": v for k, v in weight_dict.items()})
    weight_dict.update(aux_weight_dict)
    model.criterion.weight_dict = weight_dict

if mask_dino_single_loss:
    weight_dict = model.criterion.weight_dict

    for i in range(model.transformer.decoder.num_layers - 1):
        weight_dict.update({f"loss_mask_maskdino_{i}": 0})
        weight_dict.update({f"loss_dice_maskdino_{i}": 0})

    model.criterion.weight_dict = weight_dict


if mask_combine_loss:
    weight_dict = model.criterion.weight_dict

    weight_dict.update({"loss_mask_maskdino": 0})
    weight_dict.update({"loss_dice_maskdino": 0})

    model.criterion.weight_dict = weight_dict

loss_boxes_panoptic = False
if loss_boxes_panoptic:
    model.criterion.losses = ["class", "boxes_panoptic", "masks"]
