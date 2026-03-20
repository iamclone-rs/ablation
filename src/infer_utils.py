from pathlib import Path

import torch
from PIL import Image, ImageOps
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from src.sketchy_dataset import normal_transform


def canonicalize_classname(name):
    return "".join(ch for ch in name.lower() if ch.isalnum())


class FolderInferenceDataset(Dataset):
    def __init__(self, samples, max_size):
        self.samples = samples
        self.max_size = max_size
        self.transform = normal_transform()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Missing image file: {image_path}")

        with Image.open(image_path) as image_file:
            image = ImageOps.pad(
                image_file.convert("RGB"),
                size=(self.max_size, self.max_size),
            )

        image_tensor = self.transform(image)
        return image_tensor, label


def add_infer_args(parser, dataset_name):
    parser.add_argument("--root", type=str, default="", help="dataset root containing photo/ and sketch/ subfolders")
    parser.add_argument("--photo_root", type=str, default="", help="path to the photo root directory")
    parser.add_argument("--sketch_root", type=str, default="", help="path to the sketch root directory")
    parser.add_argument("--ckpt_path", type=str, required=True, help="path to the trained checkpoint")
    parser.add_argument("--dataset", type=str, default=dataset_name, help="target dataset name")
    parser.add_argument("--backbone", type=str, default="ViT-B/32")
    parser.add_argument("--n_ctx", type=int, default=2)
    parser.add_argument("--img_ctx", type=int, default=2)
    parser.add_argument("--max_size", type=int, default=224)
    parser.add_argument("--prompt_depth", type=int, default=12)
    parser.add_argument("--data_split", type=int, default=-1)
    parser.add_argument("--prec", type=str, default="fp16")
    parser.add_argument("--distill", type=str, default="cosine")
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--lambd", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--test_batch_size", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--use_adapt_sk", type=bool, default=True)
    parser.add_argument("--use_adapt_ph", type=bool, default=True)
    parser.add_argument("--use_adapt_txt", type=bool, default=True)
    parser.add_argument("--use_co_sk", type=bool, default=True)
    parser.add_argument("--use_co_ph", type=bool, default=True)
    parser.add_argument("--progress", type=bool, default=False)
    parser.add_argument("--use_subset", type=bool, default=False)
    return parser


def resolve_data_roots(args):
    if args.photo_root and args.sketch_root:
        photo_root = Path(args.photo_root)
        sketch_root = Path(args.sketch_root)
    elif args.root:
        photo_root = Path(args.root) / "photo"
        sketch_root = Path(args.root) / "sketch"
    else:
        raise ValueError("Provide either --root or both --photo_root and --sketch_root.")

    if not photo_root.exists():
        raise FileNotFoundError(f"Photo root does not exist: {photo_root}")
    if not sketch_root.exists():
        raise FileNotFoundError(f"Sketch root does not exist: {sketch_root}")

    return photo_root, sketch_root


def get_class_dir_map(root):
    class_dir_map = {}
    for path in Path(root).iterdir():
        if path.is_dir():
            class_dir_map[canonicalize_classname(path.name)] = path
    return class_dir_map


def build_samples_from_directories(photo_root, sketch_root, allowed_classnames):
    photo_dir_map = get_class_dir_map(photo_root)
    sketch_dir_map = get_class_dir_map(sketch_root)

    photo_samples = []
    sketch_samples = []
    missing_photo = []
    missing_sketch = []
    empty_classes = []

    for label, classname in enumerate(allowed_classnames):
        key = canonicalize_classname(classname)

        photo_class_dir = photo_dir_map.get(key)
        sketch_class_dir = sketch_dir_map.get(key)

        if photo_class_dir is None:
            missing_photo.append(classname)
            continue
        if sketch_class_dir is None:
            missing_sketch.append(classname)
            continue

        class_photo_samples = sorted(str(path) for path in photo_class_dir.iterdir() if path.is_file())
        class_sketch_samples = sorted(str(path) for path in sketch_class_dir.iterdir() if path.is_file())

        if not class_photo_samples or not class_sketch_samples:
            empty_classes.append(classname)
            continue

        photo_samples.extend((path, label) for path in class_photo_samples)
        sketch_samples.extend((path, label) for path in class_sketch_samples)

    if missing_photo:
        raise FileNotFoundError(f"Missing photo class folders: {missing_photo}")
    if missing_sketch:
        raise FileNotFoundError(f"Missing sketch class folders: {missing_sketch}")
    if empty_classes:
        raise ValueError(f"These classes do not contain both photo and sketch samples: {empty_classes}")

    return sketch_samples, photo_samples


def build_dataloader(samples, args):
    dataset = FolderInferenceDataset(samples=samples, max_size=args.max_size)
    return DataLoader(
        dataset=dataset,
        batch_size=args.test_batch_size,
        num_workers=args.workers,
        shuffle=False,
    )


def load_model_from_checkpoint(args, classnames):
    from src.model import ZS_SBIR

    ckpt_path = Path(args.ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    state_dict = state_dict.copy()

    skip = [
        "model.prompt_learner_photo.token_prefix",
        "model.prompt_learner_photo.token_suffix",
        "model.prompt_learner_sketch.token_prefix",
        "model.prompt_learner_sketch.token_suffix",
    ]
    for key in skip:
        state_dict.pop(key, None)

    model = ZS_SBIR(args=args, classname=classnames)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Missing keys while loading checkpoint: {missing}")
    if unexpected:
        print(f"Unexpected keys while loading checkpoint: {unexpected}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    if device.type == "cpu":
        model = model.float()
        model.model.dtype = torch.float32
        model.model.text_encoder.dtype = torch.float32

    return model, device


def extract_features(model, dataloader, classnames, image_type, device):
    features = []
    labels = []
    with torch.no_grad():
        for images, batch_labels in dataloader:
            images = images.to(device, non_blocking=True)
            batch_features = model.model.extract_feature(images, classname=classnames, type=image_type)
            features.append(batch_features.float().cpu())
            labels.append(batch_labels.cpu())

    return torch.cat(features, dim=0), torch.cat(labels, dim=0)


def evaluate_retrieval(query_features, gallery_features, query_labels, gallery_labels, p_at_k, map_at_k=0):
    from torchmetrics.functional import retrieval_average_precision, retrieval_precision

    ap = torch.zeros(len(query_features), dtype=torch.float32)
    precision = torch.zeros(len(query_features), dtype=torch.float32)

    top_k_precision = min(p_at_k, len(gallery_features))
    for idx, query_feature in enumerate(query_features):
        category = query_labels[idx].item()
        scores = F.cosine_similarity(query_feature.unsqueeze(0), gallery_features)
        target = gallery_labels == category

        if map_at_k > 0:
            top_k_map = min(map_at_k, len(gallery_features))
            ap[idx] = retrieval_average_precision(scores, target, top_k=top_k_map)
        else:
            ap[idx] = retrieval_average_precision(scores, target)

        precision[idx] = retrieval_precision(scores, target, top_k=top_k_precision)

    return ap.mean().item(), precision.mean().item()


def run_inference(args, p_at_k, map_at_k=0, allowed_classnames=None):
    photo_root, sketch_root = resolve_data_roots(args)
    classnames = list(allowed_classnames) if allowed_classnames is not None else []
    if not classnames:
        raise ValueError("allowed_classnames must be provided for across-dataset inference.")

    sketch_samples, photo_samples = build_samples_from_directories(
        photo_root=photo_root,
        sketch_root=sketch_root,
        allowed_classnames=classnames,
    )

    sketch_loader = build_dataloader(sketch_samples, args)
    photo_loader = build_dataloader(photo_samples, args)

    model, device = load_model_from_checkpoint(args, classnames)
    sketch_features, sketch_labels = extract_features(model, sketch_loader, classnames, image_type="sketch", device=device)
    photo_features, photo_labels = extract_features(model, photo_loader, classnames, image_type="photo", device=device)

    mAP, precision = evaluate_retrieval(
        query_features=sketch_features,
        gallery_features=photo_features,
        query_labels=sketch_labels,
        gallery_labels=photo_labels,
        p_at_k=p_at_k,
        map_at_k=map_at_k,
    )

    print(f"Dataset: {args.dataset}")
    print(f"Classes: {len(classnames)}")
    print(f"Sketch root: {sketch_root}")
    print(f"Photo root: {photo_root}")
    print(f"Queries: {len(sketch_features)} | Gallery: {len(photo_features)}")
    if map_at_k > 0:
        print(f"mAP@{map_at_k}: {mAP:.6f}")
    else:
        print(f"mAP@all: {mAP:.6f}")
    print(f"P@{p_at_k}: {precision:.6f}")
