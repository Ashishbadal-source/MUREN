from types import SimpleNamespace
import torch
from models import build_model

args = SimpleNamespace(
    backbone="resnet50",
    dataset_file="hico",
    dilation=False,
    position_embedding="sine",
    hidden_dim=256,
    nheads=8,
    enc_layers=6,
    dec_layers=6,
    dim_feedforward=2048,
    dropout=0.1,
    pre_norm=False,
    num_queries=64,
    num_obj_classes=80,
    num_verb_classes=117,
    aux_loss=True,
    use_matching=False,
    subject_category_id=0,
    verb_loss_type="focal",
    set_cost_class=1,
    set_cost_bbox=2.5,
    set_cost_giou=1,
    set_cost_obj_class=1,
    set_cost_verb_class=1,
    set_cost_matching=1,
    bbox_loss_coef=2.5,
    giou_loss_coef=1,
    obj_loss_coef=1,
    verb_loss_coef=2,
    matching_loss_coef=1,
    eos_coef=0.1,
    alpha=0.5,
    masks=False,
    lr_backbone=0,
    no_aux_loss=False,
    device="cpu",
)

model, criterion, postprocessors = build_model(args)
checkpoint = torch.load("pretrained/hico-best.pth", map_location="cpu")
result = model.load_state_dict(checkpoint["model"], strict=True)

model.eval()
print("Model initialized successfully")
print("Checkpoint epoch:", checkpoint.get("epoch"))
print("Weights loaded:", result)
print("Device: CPU")
