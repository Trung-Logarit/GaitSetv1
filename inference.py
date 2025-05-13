#!/usr/bin/env python3
# inference.py

import os
import argparse
import torch
import numpy as np
import pickle
from tqdm import tqdm
from model.network.gaitset import SetNet
from model.utils.data_set import DataSet

# Dùng GPU nếu có, ngược lại CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def build_dataset_from_pretr(root, resolution=64):
    seq_dirs, labels, seq_types, views = [], [], [], []
    for subj in sorted(os.listdir(root)):
        subj_path = os.path.join(root, subj)
        if not os.path.isdir(subj_path): continue
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
    encoder.eval()
    feats, views, seq_types, labels = [], [], [], []
    for i in tqdm(range(len(dataset)), desc=desc):
        data, _, view, seq_type, label = dataset[i]
        seq = data[0].values.astype('float32') / 255.0   # (F,H,W)
        tensor = torch.from_numpy(seq).unsqueeze(0).to(device)  # (1,F,H,W)
        with torch.no_grad():
            feature, _ = encoder(tensor, None)  # (1,P,M,D)
        D = feature.numel() // feature.size(0)
        f = feature.view(feature.size(0), D).cpu().numpy()
        feats.append(f[0])
        views.append(int(view))
        seq_types.append(seq_type)
        labels.append(label)
    return np.vstack(feats), views, seq_types, labels

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pretr',      type=str, required=True,
                   help='root folder của data_pretr')
    p.add_argument('--checkpoint', type=str, required=True,
                   help='encoder checkpoint (.ptm)')
    p.add_argument('--probe',      type=str, required=True,
                   help='thư mục probe, ví dụ …/001/bg-01/000')
    p.add_argument('--cache',      action='store_true',
                   help='bật cache gallery embeddings')
    args = p.parse_args()

    # 1) Load checkpoint, detect hidden_dim
    raw = torch.load(args.checkpoint, map_location=device)
    hidden_dim = next(v.shape[2] for k,v in raw.items() if k.endswith("fc_bin.0"))
    print("Detected hidden_dim:", hidden_dim)
    enc = SetNet(hidden_dim).float().to(device)
    cleaned = {k[7:]:v for k,v in raw.items()}
    enc.load_state_dict(cleaned)
    enc.eval()
    print("Loaded checkpoint:", args.checkpoint)

    # 2) Gallery: load cache nếu có, hoặc extract + save cache
    cache_path = os.path.join(args.pretr, 'gallery_cache.npz')
    if args.cache and os.path.isfile(cache_path):
        print("Loading gallery embeddings from cache:", cache_path)
        npz = np.load(cache_path, allow_pickle=True)
        gallery_feats  = npz['feats']
        gallery_views  = npz['views'].tolist()
        gallery_labels = npz['labels'].tolist()
    else:
        print("Building gallery from:", args.pretr)
        gallery_ds = build_dataset_from_pretr(args.pretr, resolution=64)
        print(f"  Gallery sequences: {len(gallery_ds)}")
        gallery_feats, gallery_views, _, gallery_labels = extract_embeddings(
            enc, gallery_ds, desc="Gallery")
        if args.cache:
            print("Saving gallery cache to:", cache_path)
            np.savez(cache_path,
                     feats=gallery_feats,
                     views=np.array(gallery_views),
                     labels=np.array(gallery_labels))

    # 3) Probe
    subj = os.path.basename(os.path.dirname(os.path.dirname(args.probe)))
    view = os.path.basename(args.probe)
    print(f"Probe → subject: {subj}, view: {view}")
    probe_ds = DataSet([[args.probe]], [subj], ['probe'], [view],
                       cache=False, resolution=64)
    probe_feats, _, _, _ = extract_embeddings(enc, probe_ds, desc="Probe ")

    # 4) Matching + confidence
    dists = np.linalg.norm(gallery_feats - probe_feats[0], axis=1)
    idx = np.argmin(dists)
    # Softmax over -distance
    sims = np.exp(-dists)
    probs = sims / sims.sum()
    conf = probs[idx] * 100

    print(f"⇒ Predicted ID: {gallery_labels[idx]} (confidence: {conf:.2f}%)")
    print(f"⇒ Predicted view: {gallery_views[idx]}°")

if __name__ == '__main__':
    main()
