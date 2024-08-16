# In small and tiny arch, remove drop path and EMA hook comparing with the
# original config
_base_ = [
    '../_base_/datasets/imagenet_bs64_swin_224.py',
    '../_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '../_base_/default_runtime.py'
]

# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='BayesDecoderVisionTransformer',
        arch='deit-tiny',
        img_size=224,
        patch_size=16,
        with_cls_token=True,
        out_type='raw',  # 
        scale=192 ** -0.5 / 0.125,
        # num_extra_tokens=4,
        # num_feature_tokens=1,
        ),
    neck=None,
    head=dict(
        type='VisionTransformerClsHead',
        num_classes=1000,
        in_channels=197,
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
    ),
    init_cfg=[
        dict(type='TruncNormal', layer='Linear', std=.02),
        dict(type='Constant', layer='LayerNorm', val=1., bias=0.),
    ],
    train_cfg=dict(augments=[
        dict(type='Mixup', alpha=0.8),
        dict(type='CutMix', alpha=1.0)
    ]),
)

# schedule settings
optim_wrapper = dict(
    paramwise_cfg=dict(
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
        custom_keys={
            '.cls_token': dict(decay_mult=0.0),
            '.pos_embed': dict(decay_mult=0.0)
        }),
    clip_grad=dict(max_norm=5.0),
)

param_scheduler = [
    dict(
        by_epoch=True,
        convert_to_iter_based=True,
        end=20,
        start_factor=0.001,
        type='LinearLR'),
    dict(begin=20, by_epoch=True, eta_min=1e-05, type='CosineAnnealingLR'),
]

randomness = dict(deterministic=True, seed=420)

# data settings
train_dataloader = dict(
    batch_size = 256,
    dataset = dict(data_root = '/root/autodl-pub/imagenet'),
)

val_dataloader = dict(
    dataset = dict(data_root = '/root/autodl-pub/imagenet'),
)

test_dataloader = dict(
    dataset = dict(data_root = '/root/autodl-pub/imagenet'),
)

visualizer = dict(
    vis_backends=[dict(type='WandbVisBackend')]
)

default_hooks = dict(
    checkpoint = dict(type='CheckpointHook', interval=1, max_keep_ckpts=5, by_epoch=True, 
                      save_best="auto", rule='less')
)