_base_ = 'grounding_dino_swin-l_pretrain_all.py'

load_from = 'https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/grounding_dino_swin-l_pretrain_all/grounding_dino_swin-l_pretrain_all-56d69e78.pth'

data_root = 'data/til/'

train_dataloader = dict(
    batch_size=1,
    num_workers=4,
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
                pipeline=_base_.train_pipeline,
                return_classes=True,
            )
        ],
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=False,
    
    dataset=dict(
        type='ODVGDataset',
        data_root=data_root,
        ann_file='annotations_val.jsonl',
        data_prefix=dict(img='val/'),
        pipeline=_base_.test_pipeline,
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

# TODO: scale LR such that LR * effective BS = constant
# Better yet, perform gradient accumulation to fix effective BS
optim_wrapper = dict(
    type='AmpOptimWrapper',
    # only devices with compute capability >=8.0 support bfloat16
    # T4 has compute capability 7.5
    dtype='float16',
    optimizer=dict(lr=0.0001),
)

max_iter = 250000
train_cfg = dict(
    _delete_=True,
    type='IterBasedTrainLoop',
    max_iters=max_iter,
    val_interval=50,
)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=False, begin=0, end=1000),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_iter,
        by_epoch=False,
        milestones=[210000],
        gamma=0.1,
    ),
]

default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=13000, max_keep_ckpts=30))
log_processor = dict(by_epoch=False)
