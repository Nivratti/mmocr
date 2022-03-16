_base_ = [
    '../../_base_/runtime_10e.py',
    '../../_base_/schedules/schedule_adam_step_600e.py',
    '../../_base_/det_models/psenet_r50_fpnf.py',
    '../../_base_/det_datasets/icdar2015.py',
    '../../_base_/det_pipelines/psenet_pipeline.py'
]

model = {{_base_.model_quad}}

train_list = {{_base_.train_list}}
test_list = {{_base_.test_list}}

train_pipeline = {{_base_.train_pipeline}}
test_pipeline_icdar2015 = {{_base_.test_pipeline_icdar2015}}

data = dict(
    samples_per_gpu=16,
    workers_per_gpu=16,
    val_dataloader=dict(samples_per_gpu=8),
    test_dataloader=dict(samples_per_gpu=8),
    train=dict(
        type='UniformConcatDataset',
        datasets=train_list,
        pipeline=train_pipeline),
    val=dict(
        type='UniformConcatDataset',
        datasets=test_list,
        pipeline=test_pipeline_icdar2015),
    test=dict(
        type='UniformConcatDataset',
        datasets=test_list,
        pipeline=test_pipeline_icdar2015))

evaluation = dict(interval=10, metric='hmean-iou')

# -------------------------
## modified -- configs/_base_/schedules/schedule_adam_step_600e.py
# optimizer
optimizer = dict(type='Adam', lr=1e-8) # 1e-7 # 1e-6 # 1e-4
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[180, 200])
total_epochs = 200