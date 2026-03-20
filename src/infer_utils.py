from pathlib import Path

import torch
from PIL import Image, ImageOps
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from src.sketchy_dataset import normal_transform


class SplitInferenceDataset(Dataset):
    def __init__(self, root, samples, max_size):
        self.root = Path(root)
        self.samples = samples
        self.max_size = max_size
        self.transform = normal_transform()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        rel_path, label = self.samples[index]
        image_path = self.root / Path(rel_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Missing file referenced by split: {image_path}")

        with Image.open(image_path) as image_file:
            image = ImageOps.pad(
                image_file.convert("RGB"),
                size=(self.max_size, self.max_size),
            )

        image_tensor = self.transform(image)
        return image_tensor, label


def add_infer_args(parser, dataset_name, default_meta_root):
    parser.add_argument("--root", type=str, required=True, help="dataset root containing image folders referenced by the split txt files")
    parser.add_argument("--ckpt_path", type=str, required=True, help="path to the trained checkpoint")
    parser.add_argument("--meta_root", type=str, default=str(default_meta_root), help="directory containing split txt files")
    parser.add_argument("--split", type=str, default="zero", choices=["zero", "train"], help="which split txt files to use")
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


def load_classnames(cname_file):
    cid_to_classname = {}
    with Path(cname_file).open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            classname, cid = line.rsplit(maxsplit=1)
            cid_to_classname[int(cid)] = classname

    return [cid_to_classname[idx] for idx in sorted(cid_to_classname.keys())]


def load_split_samples(split_file):
    samples = []
    with Path(split_file).open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            rel_path, label = line.rsplit(maxsplit=1)
            samples.append((rel_path, int(label)))
    return samples


def filter_samples_by_classnames(split_file, cname_file, allowed_classnames=None):
    all_classnames = load_classnames(cname_file)
    all_samples = load_split_samples(split_file)

    if allowed_classnames is None:
        return all_samples, all_classnames

    missing_classnames = sorted(set(allowed_classnames) - set(all_classnames))
    if missing_classnames:
        raise ValueError(f"Missing classnames in {cname_file}: {missing_classnames}")

    original_cid_to_classname = {cid: classname for cid, classname in enumerate(all_classnames)}
    classname_to_new_cid = {classname: cid for cid, classname in enumerate(allowed_classnames)}

    filtered_samples = []
    for rel_path, original_cid in all_samples:
        classname = original_cid_to_classname[original_cid]
        if classname in classname_to_new_cid:
            filtered_samples.append((rel_path, classname_to_new_cid[classname]))

    if not filtered_samples:
        raise ValueError(f"No samples left after filtering {split_file}.")

    return filtered_samples, list(allowed_classnames)


def build_dataloader(root, samples, args):
    dataset = SplitInferenceDataset(root=root, samples=samples, max_size=args.max_size)
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


def run_inference(args, sketch_prefix, photo_prefix, p_at_k, map_at_k=0, allowed_classnames=None):
    meta_root = Path(args.meta_root)
    sketch_file = meta_root / f"{sketch_prefix}_{args.split}.txt"
    photo_file = meta_root / f"{photo_prefix}_{args.split}.txt"
    cname_file = meta_root / "cname_cid_zero.txt" if args.split == "zero" else meta_root / "cname_cid.txt"

    for required_path in [meta_root, sketch_file, photo_file, cname_file]:
        if not required_path.exists():
            raise FileNotFoundError(f"Missing inference metadata file: {required_path}")

    sketch_samples, classnames = filter_samples_by_classnames(
        split_file=sketch_file,
        cname_file=cname_file,
        allowed_classnames=allowed_classnames,
    )
    photo_samples, photo_classnames = filter_samples_by_classnames(
        split_file=photo_file,
        cname_file=cname_file,
        allowed_classnames=allowed_classnames,
    )
    if classnames != photo_classnames:
        raise ValueError("Sketch and photo class orders do not match after filtering.")

    sketch_loader = build_dataloader(args.root, sketch_samples, args)
    photo_loader = build_dataloader(args.root, photo_samples, args)

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

    if map_at_k > 0:
        print(f"Dataset: {args.dataset}")
        print(f"Split: {args.split}")
        print(f"Classes: {len(classnames)}")
        print(f"Queries: {len(sketch_features)} | Gallery: {len(photo_features)}")
        print(f"mAP@{map_at_k}: {mAP:.6f}")
        print(f"P@{p_at_k}: {precision:.6f}")
    else:
        print(f"Dataset: {args.dataset}")
        print(f"Split: {args.split}")
        print(f"Classes: {len(classnames)}")
        print(f"Queries: {len(sketch_features)} | Gallery: {len(photo_features)}")
        print(f"mAP@all: {mAP:.6f}")
        print(f"P@{p_at_k}: {precision:.6f}")
