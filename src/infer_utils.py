from pathlib import Path

import torch
from PIL import Image, ImageOps
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from src.sketchy_dataset import normal_transform


def canonicalize_classname(name):
    return "".join(ch for ch in name.lower() if ch.isalnum())


def sanitize_path_string(path_string):
    return (path_string or "").strip().strip('"').strip("'")


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


def add_infer_args(parser, dataset_name, dataset_choices=None):
    parser.add_argument("--root", type=str, default="", help="dataset root containing photo/ and sketch/ subfolders")
    parser.add_argument("--photo_root", type=str, default="", help="path to the photo root directory")
    parser.add_argument("--sketch_root", type=str, default="", help="path to the sketch root directory")
    parser.add_argument("--ckpt_path", type=str, required=True, help="path to the trained checkpoint")
    parser.add_argument("--dataset", type=str, default=dataset_name, choices=dataset_choices, help="target dataset name")
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


def find_kaggle_suffix_match(path):
    kaggle_root = Path("/kaggle/input")
    if not str(path).startswith("/kaggle/input") or not kaggle_root.exists():
        return None

    tail_options = []
    parts = path.parts
    for length in [4, 3, 2]:
        if len(parts) >= length:
            tail_options.append(parts[-length:])

    matches = []
    for candidate in kaggle_root.rglob(path.name):
        if not candidate.exists():
            continue
        candidate_parts = candidate.parts
        if any(len(candidate_parts) >= len(tail) and candidate_parts[-len(tail):] == tail for tail in tail_options):
            matches.append(candidate)

    if len(matches) == 1:
        return matches[0]

    if len(matches) > 1:
        matches = sorted(matches, key=lambda item: len(item.parts))
        return matches[0]

    return None


def resolve_existing_path(path_string, label):
    sanitized = sanitize_path_string(path_string)
    path = Path(sanitized)
    if path.exists():
        return path

    kaggle_match = find_kaggle_suffix_match(path)
    if kaggle_match is not None:
        print(f"{label} not found exactly, using Kaggle match: {kaggle_match}")
        return kaggle_match

    raise FileNotFoundError(f"{label} does not exist: {sanitized!r}")


def resolve_data_roots(args):
    if args.photo_root and args.sketch_root:
        photo_root = resolve_existing_path(args.photo_root, "Photo root")
        sketch_root = resolve_existing_path(args.sketch_root, "Sketch root")
    elif args.root:
        root = Path(sanitize_path_string(args.root))
        if root.name.lower() == "photo" and root.exists():
            photo_root = root
            sketch_root = resolve_existing_path(str(root.parent / "sketch"), "Sketch root")
        elif root.name.lower() == "sketch" and root.exists():
            sketch_root = root
            photo_root = resolve_existing_path(str(root.parent / "photo"), "Photo root")
        else:
            photo_root = resolve_existing_path(str(root / "photo"), "Photo root")
            sketch_root = resolve_existing_path(str(root / "sketch"), "Sketch root")
    else:
        raise ValueError("Provide either --root or both --photo_root and --sketch_root.")

    return photo_root, sketch_root


def get_class_dir_map(root):
    class_dir_map = {}
    for path in Path(root).iterdir():
        if path.is_dir():
            class_dir_map[canonicalize_classname(path.name)] = (path.name, path)
    return class_dir_map


def list_common_classnames(photo_root, sketch_root):
    photo_dir_map = get_class_dir_map(photo_root)
    sketch_dir_map = get_class_dir_map(sketch_root)
    common_keys = sorted(set(photo_dir_map.keys()) & set(sketch_dir_map.keys()))
    return [photo_dir_map[key][0] for key in common_keys]


def resolve_requested_classnames(requested_classnames, available_classnames):
    available_map = {canonicalize_classname(classname): classname for classname in available_classnames}
    resolved = []
    missing = []

    for classname in requested_classnames:
        resolved_name = available_map.get(canonicalize_classname(classname))
        if resolved_name is None:
            missing.append(classname)
        else:
            resolved.append(resolved_name)

    if missing:
        raise FileNotFoundError(f"Missing classes in folder structure: {missing}")

    return resolved


def build_samples_from_directories(photo_root, sketch_root, label_classnames, sketch_classnames=None, photo_classnames=None):
    photo_dir_map = get_class_dir_map(photo_root)
    sketch_dir_map = get_class_dir_map(sketch_root)
    available_classnames = list_common_classnames(photo_root, sketch_root)

    resolved_label_classnames = resolve_requested_classnames(label_classnames, available_classnames)
    resolved_sketch_classnames = resolve_requested_classnames(
        sketch_classnames or resolved_label_classnames,
        available_classnames,
    )
    resolved_photo_classnames = resolve_requested_classnames(
        photo_classnames or resolved_label_classnames,
        available_classnames,
    )

    label_map = {
        canonicalize_classname(classname): label
        for label, classname in enumerate(resolved_label_classnames)
    }

    def collect_samples(requested_classnames, dir_map):
        samples = []
        empty_classes = []
        for classname in requested_classnames:
            key = canonicalize_classname(classname)
            _, class_dir = dir_map[key]
            class_samples = sorted(str(path) for path in class_dir.iterdir() if path.is_file())
            if not class_samples:
                empty_classes.append(classname)
                continue
            label = label_map[key]
            samples.extend((path, label) for path in class_samples)

        if empty_classes:
            raise ValueError(f"These classes do not contain samples: {empty_classes}")

        return samples

    sketch_samples = collect_samples(resolved_sketch_classnames, sketch_dir_map)
    photo_samples = collect_samples(resolved_photo_classnames, photo_dir_map)
    return sketch_samples, photo_samples, resolved_label_classnames


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


def print_metrics(args, classnames, sketch_root, photo_root, query_count, gallery_count, p_at_k, mAP, precision, map_at_k=0, query_class_count=None):
    print(f"Dataset: {args.dataset}")
    print(f"Classes: {len(classnames)}")
    if query_class_count is not None:
        print(f"Query classes: {query_class_count}")
    print(f"Sketch root: {sketch_root}")
    print(f"Photo root: {photo_root}")
    print(f"Queries: {query_count} | Gallery: {gallery_count}")
    if map_at_k > 0:
        print(f"mAP@{map_at_k}: {mAP:.6f}")
    else:
        print(f"mAP@all: {mAP:.6f}")
    print(f"P@{p_at_k}: {precision:.6f}")


def run_inference(args, p_at_k, map_at_k=0, allowed_classnames=None):
    photo_root, sketch_root = resolve_data_roots(args)
    if not allowed_classnames:
        raise ValueError("allowed_classnames must be provided for across-dataset inference.")

    sketch_samples, photo_samples, classnames = build_samples_from_directories(
        photo_root=photo_root,
        sketch_root=sketch_root,
        label_classnames=allowed_classnames,
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

    print_metrics(
        args=args,
        classnames=classnames,
        sketch_root=sketch_root,
        photo_root=photo_root,
        query_count=len(sketch_features),
        gallery_count=len(photo_features),
        p_at_k=p_at_k,
        mAP=mAP,
        precision=precision,
        map_at_k=map_at_k,
    )


def run_gzs_inference(args, unseen_classnames, p_at_k, map_at_k=0):
    photo_root, sketch_root = resolve_data_roots(args)
    all_classnames = list_common_classnames(photo_root, sketch_root)
    query_classnames = resolve_requested_classnames(unseen_classnames, all_classnames)

    sketch_samples, photo_samples, classnames = build_samples_from_directories(
        photo_root=photo_root,
        sketch_root=sketch_root,
        label_classnames=all_classnames,
        sketch_classnames=query_classnames,
        photo_classnames=all_classnames,
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

    print_metrics(
        args=args,
        classnames=classnames,
        sketch_root=sketch_root,
        photo_root=photo_root,
        query_count=len(sketch_features),
        gallery_count=len(photo_features),
        p_at_k=p_at_k,
        mAP=mAP,
        precision=precision,
        map_at_k=map_at_k,
        query_class_count=len(query_classnames),
    )
