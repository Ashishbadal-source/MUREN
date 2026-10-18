from types import SimpleNamespace
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw
from models import build_model

# 1. Configuration: same HICO model used for the checkpoint
args = SimpleNamespace(
    backbone="resnet50",
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
    dataset_file="hico",
    device="cpu",
)

device = torch.device("cpu")

# 2. Load pretrained model
print("Building MUREN model...")
model, criterion, postprocessors = build_model(args)

checkpoint = torch.load("pretrained/hico-best.pth", map_location="cpu")
model.load_state_dict(checkpoint["model"], strict=True)
model.to(device).eval()
print("Pretrained weights loaded.")

# 3. Load and preprocess image
image_path = Path("test4.jpg")
if not image_path.exists():
    raise FileNotFoundError("Place test4.jpg in the MUREN folder.")

image = Image.open(image_path).convert("RGB")
original_w, original_h = image.size

new_h = 800
new_w = int(original_w * new_h / original_h)
if new_w > 1333:
    new_w = 1333
    new_h = int(original_h * new_w / original_w)

image_resized = image.resize((new_w, new_h), Image.Resampling.BILINEAR)

transform = T.Compose([
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
tensor = transform(image_resized).unsqueeze(0).to(device)

# 4. Inference
print("Running inference on CPU...")
with torch.no_grad():
    outputs = model(tensor)
    # Debug raw boxes for the suspicious prediction
query_id = 25

raw_sub = outputs["pred_sub_boxes"][0, query_id]
raw_obj = outputs["pred_obj_boxes"][0, query_id]
obj_logits = outputs["pred_obj_logits"][0, query_id]
obj_probs = obj_logits.softmax(dim=-1)

top_scores, top_ids = obj_probs[:-1].topk(5)

print("\n--- TOP OBJECT CLASSES FOR QUERY ---")
for class_id, class_score in zip(top_ids.tolist(), top_scores.tolist()):
    print("Class index:", class_id, "Score:", round(class_score, 4))

print("\n--- RAW BOX DEBUG ---")
print("Query ID:", query_id)
print("Raw subject box (normalized cx, cy, w, h):",
      raw_sub.tolist())
print("Raw object box (normalized cx, cy, w, h):",
      raw_obj.tolist())
print("Original image size (W, H):",
      original_w, original_h)
print("Resized image size (W, H):",
      new_w, new_h)

target_sizes = torch.tensor([[original_h, original_w]])
results = postprocessors["hoi"](outputs, target_sizes)[0]

print("\n--- Raw model output shapes ---")
for key, value in outputs.items():
    if torch.is_tensor(value):
        print(key, tuple(value.shape))

print("\n--- Postprocessed result shapes ---")
for key, value in results.items():
    if torch.is_tensor(value):
        print(key, tuple(value.shape))
        
# 5. Visualize top human-object interaction candidates
# draw = ImageDraw.Draw(image.copy())
output_image = image.copy()
draw = ImageDraw.Draw(output_image)

verb_scores = results["verb_scores"]
sub_boxes = results["boxes"][:args.num_queries]
obj_boxes = results["boxes"][args.num_queries:]
obj_scores = results["obj_scores"]
labels = results["labels"]

object_names = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant",
    "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
]


sub_ids = results["sub_ids"]
obj_ids = results["obj_ids"]


# coco_names = [
#     "person", "bicycle", "car", "motorcycle", "airplane", "bus",
#     "train", "truck", "boat", "traffic light", "fire hydrant",
#     "stop sign", "parking meter", "bench", "bird", "cat", "dog",
#     "horse", "sheep", "cow", "elephant", "bear", "zebra",
#     "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase",
#     "frisbee", "skis", "snowboard", "sports ball", "kite",
#     "baseball bat", "baseball glove", "skateboard", "surfboard",
#     "tennis racket", "bottle", "wine glass", "cup", "fork", "knife",
#     "spoon", "bowl", "banana", "apple", "sandwich", "orange",
#     "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
#     "chair", "couch", "potted plant", "bed", "dining table",
#     "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
#     "cell phone", "microwave", "oven", "toaster", "sink",
#     "refrigerator", "book", "clock", "vase", "scissors",
#     "teddy bear", "hair drier", "toothbrush"
# ]


print("Unique predicted labels:", sorted(set(labels.tolist())))
print("Object IDs:", obj_ids[:10].tolist())
print("Object label IDs:", labels[obj_ids[:10]].tolist())



print("labels shape:", labels.shape)
print("sub_ids shape:", sub_ids.shape)
print("obj_ids shape:", obj_ids.shape)
print("First labels:", labels[:10].tolist())
print("First subject IDs:", sub_ids[:10].tolist())
print("First object IDs:", obj_ids[:10].tolist())

verb_names = [
    "adjust", "assemble", "block", "blow", "board", "break",
    "brush_with", "buy", "carry", "catch", "chase", "check",
    "clean", "control", "cook", "cut", "cut_with", "direct",
    "drag", "dribble", "drink_with", "drive", "dry", "eat",
    "eat_at", "exit", "feed", "fill", "flip", "flush", "fly",
    "greet", "grind", "groom", "herd", "hit", "hold", "hop_on",
    "hose", "hug", "hunt", "inspect", "install", "jump", "kick",
    "kiss", "lasso", "launch", "lick", "lie_on", "lift", "light",
    "load", "lose", "make", "milk", "move", "no_interaction",
    "open", "operate", "pack", "paint", "park", "pay", "peel",
    "pet", "pick", "pick_up", "point", "pour", "pull", "push",
    "race", "read", "release", "repair", "ride", "row", "run",
    "sail", "scratch", "serve", "set", "shear", "sign", "sip",
    "sit_at", "sit_on", "slide", "smell", "spin", "squeeze",
    "stab", "stand_on", "stand_under", "stick", "stir",
    "stop_at", "straddle", "swing", "tag", "talk_on", "teach",
    "text_on", "throw", "tie", "toast", "train", "turn",
    "type_on", "walk", "wash", "watch", "wave", "wear",
    "wield", "zip"
]

assert len(verb_names) == args.num_verb_classes


# Show highest-scoring candidate interactions
# Show filtered interaction candidates
candidates = []

for query_id in range(args.num_queries):
    verb_id = int(verb_scores[query_id].argmax())
    score = float(verb_scores[query_id, verb_id])
    object_score = float(obj_scores[query_id])

    # Initial thresholds for visual testing, not benchmark defaults
    if score < 0.20 or object_score < 0.50:
        continue

    candidates.append((score, query_id, verb_id, object_score))

candidates.sort(reverse=True)

shown = 0
used = set()

for score, query_id, verb_id, object_score in candidates:
    if query_id in used:
        continue

    used.add(query_id)
    # Skip near-duplicate person-object predictions
    # if shown > 0:
    #     continue
    object_class_id = int(labels[obj_ids[query_id]])
    object_name = object_names[object_class_id]
    # Show top 3 predicted verbs for this interaction
    top_scores, top_ids = verb_scores[query_id].topk(3)

    print("Top 3 predicted actions:")
    for rank, (vid, vscore) in enumerate(
        zip(top_ids.tolist(), top_scores.tolist()), start=1
    ):
        print(
            f"  {rank}. {verb_names[vid]} "
            f"(score={vscore:.3f}, class_id={vid})"
        )

    print("-" * 40)

    sb = sub_boxes[query_id].tolist()
    ob = obj_boxes[query_id].tolist()

    print("Query:", query_id)
    print("Subject box:", sb)
    print("Object box:", ob)
    print("Object class index:", object_class_id)
    print("Object name:", object_name)
    print("Object score:", object_score)
    print("Verb class index:", verb_id)
    print("Verb score:", score)
    print("-" * 40)

    def clip_box(box):
        x1, y1, x2, y2 = box
        return [
            max(0, min(original_w - 1, x1)),
            max(0, min(original_h - 1, y1)),
            max(0, min(original_w - 1, x2)),
            max(0, min(original_h - 1, y2)),
        ]

    sb = clip_box(sb)
    ob = clip_box(ob)

    draw.rectangle(sb, outline="lime", width=3)
    draw.rectangle(ob, outline="red", width=3)

    label = f"{object_name}: {verb_names[verb_id]} ({score:.2f})"
    draw.text((sb[0], max(0, sb[1] - 18)), label, fill="lime")

    print(
        f"Candidate {shown + 1}: person -> {object_name} -> "
        f"{verb_names[verb_id]}, interaction={score:.3f}, "
        f"object={object_score:.3f}"
    )

    shown += 1
    if shown >= 5:
        break

output_image.save("output.jpg")
print(f"Saved output.jpg; displayed {shown} candidate interactions.")
print("Note: verb_id is a class index, not the human-readable verb name.")
