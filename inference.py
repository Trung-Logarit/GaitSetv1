#!/usr/bin/env python3
# inference.py

import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
from model.network.gaitset import SetNet
from model.utils.data_set import DataSet

# Tự động dùng GPU nếu có, ngược lại dùng CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def build_dataset_from_pretr(root, resolution=64):
    """
    Quét thư mục:
      root/<subject_id>/<seq_type>/<view>/
    trả về DataSet chứa tất cả sequences.
    """
    seq_dirs, labels, seq_types, views = [], [], [], []
    for subj in sorted(os.listdir(root)):
        subj_path = os.path.join(root, subj)
        if not os.path.isdir(subj_path):
            continue
        for seq_type in sorted(os.listdir(subj_path)):
            seq_type_path = os.path.join(subj_path, seq_type)
            for view in sorted(os.listdir(seq_type_path)):
                folder = os.path.join(seq_type_path, view)
                if os.path.isdir(folder):
                    seq_dirs.append([folder])
                    labels.append(subj)
                    seq_types.append(seq_type)
                    views.append(view)
    return DataSet(seq_dirs, labels, seq_types, views,
                   cache=False, resolution=resolution)

def extract_embeddings(encoder, dataset, desc="Embedding"):
    """
    Dùng encoder để tính embedding cho toàn bộ dataset.
    Trả về:
      feats: (N, D) numpy array
      views, seq_types, labels: list độ dài N
    Hiển thị tiến độ với tqdm có mô tả desc.
    """
    encoder.eval()
    feats, views, seq_types, labels = [], [], [], []

    for i in tqdm(range(len(dataset)), desc=desc):
        data, _, view, seq_type, label = dataset[i]
        # data[0] là xarray với dims [frame, img_y, img_x]
        seq = data[0].values.astype('float32') / 255.0   # (F, H, W)

        # Tạo tensor (1, F, H, W)
        tensor = torch.from_numpy(seq).unsqueeze(0).to(device)

        with torch.no_grad():
            feature, _ = encoder(tensor, None)            # feature: (1, P, M, D)

        # flatten thành (1, D')
        D = feature.numel() // feature.size(0)
        f = feature.view(feature.size(0), D).cpu().numpy()

        feats.append(f)
        views.append(int(view))
        seq_types.append(seq_type)
        labels.append(label)

    feats = np.vstack(feats)  # (N, D')
    return feats, views, seq_types, labels

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretr',      type=str, required=True,
                        help='root folder của data_pretr')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='đường dẫn tới file encoder checkpoint (.ptm)')
    parser.add_argument('--probe',      type=str, required=True,
                        help='thư mục chứa sequence probe, ví dụ …/001/bg-01/000')
    args = parser.parse_args()

    # 1) Đọc checkpoint để detect hidden_dim
    raw = torch.load(args.checkpoint, map_location=device)
    hidden_dim = None
    for k, v in raw.items():
        if k.endswith("fc_bin.0"):
            hidden_dim = v.shape[2]
            break
    if hidden_dim is None:
        raise RuntimeError("Không tìm thấy fc_bin.0 để xác định hidden_dim")
    print("Detected hidden_dim:", hidden_dim)

    # 2) Khởi tạo encoder và load weights
    enc = SetNet(hidden_dim).float().to(device)
    cleaned = {}
    for k, v in raw.items():
        nk = k[7:] if k.startswith("module.") else k
        cleaned[nk] = v
    enc.load_state_dict(cleaned)
    enc.eval()
    print("Loaded checkpoint:", args.checkpoint)

    # 3) Build gallery
    print("Building gallery from:", args.pretr)
    gallery_ds = build_dataset_from_pretr(args.pretr, resolution=64)
    print(f"Gallery sequences: {len(gallery_ds)}")
    gallery_feats, gallery_views, _, gallery_labels = extract_embeddings(
        enc, gallery_ds, desc="Embedding gallery"
    )

    # 4) Build probe
    subj = os.path.basename(os.path.dirname(os.path.dirname(args.probe)))
    view = os.path.basename(args.probe)
    print(f"Probe folder => subject: {subj}, view: {view}")
    probe_ds = DataSet([[args.probe]], [subj], ['probe'], [view],
                       cache=False, resolution=64)
    probe_feats, _, _, _ = extract_embeddings(
        enc, probe_ds, desc="Embedding probe  "
    )

    # 5) So khớp Euclid
    dists = np.linalg.norm(gallery_feats - probe_feats[0:1], axis=1)
    idx = np.argmin(dists)
    print(f"⇒ Predicted ID: {gallery_labels[idx]}")
    print(f"⇒ Predicted view: {gallery_views[idx]}°")

if __name__ == '__main__':
    main()
