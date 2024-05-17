_base_ = 'grounding_dino_swin-l_pretrain_all.py'

load_from = 'https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/grounding_dino_swin-l_pretrain_all/grounding_dino_swin-l_pretrain_all-56d69e78.pth'

data_root = 'data/til/'

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Albu',
        transforms=[dict(
            type='GaussNoise',
            var_limit=2500,
            mean=0,
            per_channel=True,
            p=0.5,
        )],
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
        ),
    ),
    dict(type='RandomFlip', direction=['horizontal', 'vertical'], prob=[0.5, 0.5]),
    dict(
        type='RandomAffine',
        max_rotate_degree=15,
        max_translate_ratio=0.1,
        scaling_ratio_range=(0.8, 1.2),
        max_shear_degree=15,
    ),
    dict(
        type='Albu',
        transforms=[dict(
            type='RandomBrightnessContrast',
            brightness_limit=(-0.2, 0.2),
            contrast_limit=(-0.2, 0.2),
            p=0.2,
        )],
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
        ),
    ),
    dict(type='RandomChoice', transforms=[
        # choose between no transform, MixUp, and Mosaic
        [],
        [dict(
            type='MixUp',
            img_scale=(870, 1520),
            ratio_range=(0.5, 1.5),
            flip_ratio=0.5,
            pad_val=114,
            max_iters=15,
        )],
        [dict(
            type='Mosaic',
            img_scale=(870, 1520),
            center_ratio_range=(0.5, 1.5),
            pad_val=114,
        )],
    ]),
    dict(
        type='RandomChoice',
        transforms=[
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ],
            [
                dict(
                    type='RandomChoiceResize',
                    # The radio of all image in train dataset < 7
                    # follow the original implement
                    scales=[(400, 4200), (500, 4200), (600, 4200)],
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(384, 600),
                    allow_negative_crop=True),
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ]
        ]),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(
        type='RandomSamplingNegPos',
        tokenizer_name=_base_.lang_model_name,
        num_sample_negative=85,
        max_tokens=256),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction', 'text',
                   'custom_entities', 'tokens_positive', 'dataset_mode'))
]

train_dataloader = dict(
    batch_size=2,
    num_workers=32,
    persistent_workers=True,
    
    # overwrite CustomSampleSizeSampler that expects to be sampling from 13 datasets
    sampler=dict(
        _delete_=True,
        type='InfiniteSampler',
        shuffle=True,
    ),
    
    dataset=dict(
        _delete_=True,
        type='ConcatDataset',
        datasets=[
            dict(
                type='ODVGDataset',
                data_root=data_root,
                ann_file='annotations_train.jsonl',
                label_map_file=None,
                data_prefix=dict(img='train/'),
                pipeline=train_pipeline,
                return_classes=True,
            )
        ],
    )
)

test_pipeline = [
    dict(
        type='LoadImageFromFile', backend_args=None,
        imdecode_backend='pillow'),
    dict(
        type='FixScaleResize',
        scale=(800, 1333),
        keep_ratio=True,
        backend='pillow'),
    dict(type='LoadTextAnnotations'),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'text', 'custom_entities',
                   'tokens_positive'))
]

val_dataloader = dict(
    batch_size=1,
    num_workers=32,
    persistent_workers=False,
    
    dataset=dict(
        type='ODVGDataset',
        data_root=data_root,
        ann_file='annotations_val.jsonl',
        data_prefix=dict(img='val/'),
        pipeline=test_pipeline,
        return_classes=True,
    )
)
test_dataloader = val_dataloader

val_evaluator = dict(
    _delete_=True,
    type='DumpODVGResults',
    outfile_path=data_root + 'out.json',
    img_prefix=data_root + 'val/',
    score_thr=0.4,
    nms_thr=0.5,
)
test_evaluator = val_evaluator

optim_wrapper = dict(
    type='AmpOptimWrapper',
    dtype='float16',
    optimizer=dict(
        lr=1e-4,
    ),
)

max_iter = 20000
train_cfg = dict(
    _delete_=True,
    type='IterBasedTrainLoop',
    max_iters=max_iter,
    val_interval=1000,
)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_iter,
        by_epoch=False,
        milestones=[5000],
        gamma=0.1)
]

default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=1000, max_keep_ckpts=100))
log_processor = dict(by_epoch=False)
