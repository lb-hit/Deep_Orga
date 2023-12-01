train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=20, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
param_scheduler = [
    dict(
        type='mmdet.QuadraticWarmupLR',
        by_epoch=True,
        begin=0,
        end=5,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        eta_min=0.0005,
        begin=5,
        T_max=15,
        end=15,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(type='ConstantLR', by_epoch=True, factor=1, begin=15, end=20)
]
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005, nesterov=True),
    paramwise_cfg=dict(norm_decay_mult=0.0, bias_decay_mult=0.0))
auto_scale_lr = dict(enable=False, base_batch_size=64)
default_scope = 'mmdet'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook', interval=1, max_keep_ckpts=1, save_best='auto'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'))
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend')],
    name='visualizer')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'
load_from = None
resume = False
tta_model = dict(
    type='DetTTAModel',
    tta_cfg=dict(nms=dict(type='nms', iou_threshold=0.65), max_per_img=100))
img_scales = [(640, 640), (320, 320), (960, 960)]
tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[[{
            'type': 'Resize',
            'scale': (640, 640),
            'keep_ratio': True
        }, {
            'type': 'Resize',
            'scale': (320, 320),
            'keep_ratio': True
        }, {
            'type': 'Resize',
            'scale': (960, 960),
            'keep_ratio': True
        }],
                    [{
                        'type': 'RandomFlip',
                        'prob': 1.0
                    }, {
                        'type': 'RandomFlip',
                        'prob': 0.0
                    }],
                    [{
                        'type': 'Pad',
                        'pad_to_square': True,
                        'pad_val': {
                            'img': (114.0, 114.0, 114.0)
                        }
                    }], [{
                        'type': 'LoadAnnotations',
                        'with_bbox': True
                    }],
                    [{
                        'type':
                        'PackDetInputs',
                        'meta_keys':
                        ('img_id', 'img_path', 'ori_shape', 'img_shape',
                         'scale_factor', 'flip', 'flip_direction')
                    }]])
]
img_scale = (640, 640)
model = dict(
    type='YOLOX',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        pad_size_divisor=32,
        batch_augments=[
            dict(
                type='BatchSyncRandomResize',
                random_size_range=(480, 800),
                size_divisor=32,
                interval=10)
        ]),
    neck=[
        dict(
            type='YOLOXPAFPN',
            in_channels=[128, 256, 512],
            out_channels=128,
            num_csp_blocks=1,
            use_depthwise=False,
            upsample_cfg=dict(scale_factor=2, mode='nearest'),
            norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg=dict(type='Swish')),
        dict(
            type='BFP',
            in_channels=128,
            num_levels=3,
            refine_level=2,
            refine_type='non_local')
    ],
    bbox_head=dict(
        type='YOLOXHead',
        num_classes=1,
        in_channels=128,
        feat_channels=128,
        stacked_convs=2,
        strides=(8, 16, 32),
        use_depthwise=False,
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='Swish'),
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='sum',
            loss_weight=1.0),
        loss_bbox=dict(
            type='IoULoss',
            mode='square',
            eps=1e-16,
            reduction='sum',
            loss_weight=5.0),
        loss_obj=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='sum',
            loss_weight=1.0),
        loss_l1=dict(
            type='BalancedL1Loss',
            reduction='mean',
            loss_weight=1.0,
            alpha=0.5,
            gamma=1.5,
            beta=1.0)),
    train_cfg=dict(assigner=dict(type='SimOTAAssigner', center_radius=2.5)),
    test_cfg=dict(score_thr=0.01, nms=dict(type='nms', iou_threshold=0.65)),
    backbone=dict(
        type='mmpretrain.RepVGG',
        arch='yolox-pai-small',
        add_ppf=True,
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.03),
        out_indices=(1, 2, 3)))
data_root = 'D:/博士课题/工作站代码及数据集备份/lb_code/21.yolo/mmyolo/data/organoid/'
dataset_type = 'CocoDataset'
backend_args = None
train_pipeline = [
    dict(type='Mosaic', img_scale=(640, 640), pad_val=114.0),
    dict(type='RandomAffine', scaling_ratio_range=(1, 1), border=(-320, -320)),
    dict(
        type='MixUp',
        img_scale=(640, 640),
        ratio_range=(0.8, 1.6),
        pad_val=114.0),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    dict(
        type='Pad',
        pad_to_square=True,
        pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type='PackDetInputs')
]
train_dataset = dict(
    type='MultiImageMixDataset',
    dataset=dict(
        type='CocoDataset',
        data_root='D:/博士课题/工作站代码及数据集备份/lb_code/21.yolo/mmyolo/data/organoid/',
        ann_file='annotations/train.json',
        data_prefix=dict(img='images/'),
        pipeline=[
            dict(type='LoadImageFromFile', backend_args=None),
            dict(type='LoadAnnotations', with_bbox=True)
        ],
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        backend_args=None,
        metainfo=dict(classes='object', palette=[(255, 0, 0)])),
    pipeline=[
        dict(type='Mosaic', img_scale=(640, 640), pad_val=114.0),
        dict(
            type='RandomAffine',
            scaling_ratio_range=(1, 1),
            border=(-320, -320)),
        dict(
            type='MixUp',
            img_scale=(640, 640),
            ratio_range=(0.8, 1.6),
            pad_val=114.0),
        dict(type='YOLOXHSVRandomAug'),
        dict(type='RandomFlip', prob=0.5),
        dict(type='Resize', scale=(640, 640), keep_ratio=True),
        dict(
            type='Pad',
            pad_to_square=True,
            pad_val=dict(img=(114.0, 114.0, 114.0))),
        dict(
            type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
        dict(type='PackDetInputs')
    ])
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    dict(
        type='Pad',
        pad_to_square=True,
        pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='MultiImageMixDataset',
        dataset=dict(
            type='CocoDataset',
            data_root=
            'D:/博士课题/工作站代码及数据集备份/lb_code/21.yolo/mmyolo/data/organoid/',
            ann_file='annotations/train.json',
            data_prefix=dict(img='images/'),
            pipeline=[
                dict(type='LoadImageFromFile', backend_args=None),
                dict(type='LoadAnnotations', with_bbox=True)
            ],
            filter_cfg=dict(filter_empty_gt=False, min_size=32),
            backend_args=None,
            metainfo=dict(classes='object', palette=[(255, 0, 0)])),
        pipeline=[
            dict(type='Mosaic', img_scale=(640, 640), pad_val=114.0),
            dict(
                type='RandomAffine',
                scaling_ratio_range=(1, 1),
                border=(-320, -320)),
            dict(
                type='MixUp',
                img_scale=(640, 640),
                ratio_range=(0.8, 1.6),
                pad_val=114.0),
            dict(type='YOLOXHSVRandomAug'),
            dict(type='RandomFlip', prob=0.5),
            dict(type='Resize', scale=(640, 640), keep_ratio=True),
            dict(
                type='Pad',
                pad_to_square=True,
                pad_val=dict(img=(114.0, 114.0, 114.0))),
            dict(
                type='FilterAnnotations',
                min_gt_bbox_wh=(1, 1),
                keep_empty=False),
            dict(type='PackDetInputs')
        ]))
val_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        data_root='D:/博士课题/工作站代码及数据集备份/lb_code/21.yolo/mmyolo/data/organoid/',
        ann_file='annotations/test.json',
        data_prefix=dict(img='images/'),
        test_mode=True,
        pipeline=[
            dict(type='LoadImageFromFile', backend_args=None),
            dict(type='Resize', scale=(640, 640), keep_ratio=True),
            dict(
                type='Pad',
                pad_to_square=True,
                pad_val=dict(img=(114.0, 114.0, 114.0))),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                type='PackDetInputs',
                meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                           'scale_factor'))
        ],
        backend_args=None,
        metainfo=dict(classes='object', palette=[(255, 0, 0)])))
test_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        data_root='D:/博士课题/工作站代码及数据集备份/lb_code/21.yolo/mmyolo/data/organoid/',
        ann_file='annotations/test.json',
        data_prefix=dict(img='images/'),
        test_mode=True,
        pipeline=[
            dict(type='LoadImageFromFile', backend_args=None),
            dict(type='Resize', scale=(640, 640), keep_ratio=True),
            dict(
                type='Pad',
                pad_to_square=True,
                pad_val=dict(img=(114.0, 114.0, 114.0))),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                type='PackDetInputs',
                meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                           'scale_factor'))
        ],
        backend_args=None,
        metainfo=dict(classes='object', palette=[(255, 0, 0)])))
val_evaluator = dict(
    type='CocoMetric',
    ann_file=
    'D:/博士课题/工作站代码及数据集备份/lb_code/21.yolo/mmyolo/data/organoid/annotations/test.json',
    metric='bbox',
    backend_args=None)
test_evaluator = dict(
    type='CocoMetric',
    ann_file=
    'D:/博士课题/工作站代码及数据集备份/lb_code/21.yolo/mmyolo/data/organoid/annotations/test.json',
    metric='bbox',
    backend_args=None)
max_epochs = 20
num_last_epochs = 5
interval = 1
base_lr = 0.01
custom_hooks = [
    dict(type='YOLOXModeSwitchHook', num_last_epochs=5, priority=48),
    dict(type='SyncNormHook', priority=48),
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        update_buffers=True,
        priority=49)
]
custom_imports = dict(
    imports=['mmpretrain.models'], allow_failed_imports=False)
metainfo = dict(classes='object', palette=[(255, 0, 0)])
launcher = 'none'
work_dir = './work_dirs\\#O1_4yolox_s_8xb8-300e_coco_RepVGG_BFP_loss-balance-L1-0722'
