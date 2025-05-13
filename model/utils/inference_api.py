# file: model/utils/inference_api.py
import os
import numpy as np
import torch
from model.network.gaitset import SetNet
from model.utils.data_set import DataSet

def init_model(checkpoint_path: str, pretr_path: str, resolution: int = 64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw = torch.load(checkpoint_path, map_location=device)
    hidden_dim = next(v.shape[2] for k,v in raw.items() if k.endswith("fc_bin.0"))

    enc = SetNet(hidden_dim).float().to(device)
    cleaned = {k[7:]:v for k,v in raw.items() if k.startswith("module.")}
    if not cleaned:
        cleaned = raw
    enc.load_state_dict(cleaned)
    enc.eval()

    # build gallery chỉ từ các thư mục con
    seq_dirs, labels, seq_types, views = [], [], [], []
    for subj in sorted(os.listdir(pretr_path)):
        subj_path = os.path.join(pretr_path, subj)
        if not os.path.isdir(subj_path):
            continue
        for st in sorted(os.listdir(subj_path)):
            st_path = os.path.join(subj_path, st)
            if not os.path.isdir(st_path):
                continue
            for vw in sorted(os.listdir(st_path)):
                folder = os.path.join(st_path, vw)
                if not os.path.isdir(folder):
                    continue
                seq_dirs.append([folder])
                labels.append(subj)
                seq_types.append(st)
                views.append(vw)

    gallery_ds = DataSet(seq_dirs, labels, seq_types, views,
                         cache=False, resolution=resolution)
    feats, views, _, labels = extract_embeddings(enc, gallery_ds)
    return {
        "encoder": enc,
        "device": device,
        "gallery_feats": feats,
        "gallery_views": views,
        "gallery_labels": labels,
        "resolution": resolution
    }

def extract_embeddings(encoder, dataset: DataSet):
    encoder.eval()
    feats, views, seq_types, labels = [], [], [], []
    device = next(encoder.parameters()).device

    for i in range(len(dataset)):
        data, _, view, seq_type, label = dataset[i]
        arr = data[0].values if hasattr(data[0], "values") else data[0]
        seq = arr.astype("float32") / 255.0
        tensor = torch.from_numpy(seq).unsqueeze(0).to(device)
        with torch.no_grad():
            feat, _ = encoder(tensor, None)
        D = feat.numel() // feat.size(0)
        f = feat.view(feat.size(0), D).cpu().numpy()[0]
        feats.append(f)
        views.append(int(view))
        seq_types.append(seq_type)
        labels.append(label)

    return np.vstack(feats), views, seq_types, labels

def predict_from_folder(model_dict: dict, probe_folder: str):
    enc        = model_dict["encoder"]
    feats      = model_dict["gallery_feats"]
    views      = model_dict["gallery_views"]
    labels     = model_dict["gallery_labels"]
    device     = model_dict["device"]
    resolution = model_dict["resolution"]

    ds = DataSet([[probe_folder]],
                 [os.path.basename(os.path.dirname(os.path.dirname(probe_folder)))],
                 ["probe"],
                 [os.path.basename(probe_folder)],
                 cache=False,
                 resolution=resolution)
    data, _, _, _, _ = ds[0]
    arr = data[0].values if hasattr(data[0], "values") else data[0]
    seq = arr.astype("float32") / 255.0
    tensor = torch.from_numpy(seq).unsqueeze(0).to(device)
    with torch.no_grad():
        feat, _ = enc(tensor, None)
    D = feat.numel() // feat.size(0)
    probe_f = feat.view(feat.size(0), D).cpu().numpy()[0]

    dists = np.linalg.norm(feats - probe_f, axis=1)
    idx = int(np.argmin(dists))
    return labels[idx], views[idx]
