import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import ee
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import AutoImageProcessor, MobileViTModel

from get_meta_feats import (
    batch_average_ndvi_for_year,
    batch_average_precip_last_year,
    batch_average_temp_summer_winter,
)


class ImageMetaRegressor(nn.Module):
    def __init__(
        self,
        meta_dim: int,
        vit_output_dim: int = 640,
        meta_emb_dim: int = 64,
        fusion_hidden: int = 128,
    ):
        super().__init__()

        self.backbone = MobileViTModel.from_pretrained("apple/mobilevit-small")

        self.meta_mlp = nn.Sequential(
            nn.Linear(meta_dim, meta_emb_dim),
            nn.ReLU(),
            nn.Linear(meta_emb_dim, meta_emb_dim),
            nn.ReLU(),
        )

        fusion_in = vit_output_dim + meta_emb_dim
        self.reg_head = nn.Sequential(
            nn.Linear(fusion_in, fusion_hidden),
            nn.ReLU(),
            nn.Linear(fusion_hidden, 1),
        )

    def forward(self, pixel_values: torch.Tensor, meta_features: torch.Tensor) -> torch.Tensor:
        img_emb = self.backbone(pixel_values=pixel_values).pooler_output
        meta_emb = self.meta_mlp(meta_features)
        fused = torch.cat([img_emb, meta_emb], dim=-1)
        pred = self.reg_head(fused)
        return pred.squeeze(-1)


class TreelineDataset(Dataset):
    def __init__(self, image_paths: Sequence[Path], meta_features: np.ndarray, targets: np.ndarray):
        self.image_paths = list(image_paths)
        self.meta_features = torch.tensor(meta_features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        image = Image.open(self.image_paths[idx]).convert("RGB")
        return {
            "image": image,
            "meta": self.meta_features[idx],
            "target": self.targets[idx],
        }


def _build_image_candidates(
    feature_id: object,
    section_id: object,
    past_year_used: object,
) -> List[str]:
    stems = [
        f"tile_{feature_id}_{section_id}_past_{past_year_used}",
        f"tile_{feature_id}_{section_id}_old_{past_year_used}",
    ]
    exts = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ""]
    names: List[str] = []
    for stem in stems:
        for ext in exts:
            names.append(f"{stem}{ext}")
    return names


def _resolve_image_path(row: pd.Series, images_dir: Path) -> Optional[Path]:
    candidates = _build_image_candidates(
        row["feature_id"],
        row["section_id"],
        int(float(row["past_year_used"])),
    )
    for name in candidates:
        p = images_dir / name
        if p.exists() and p.is_file():
            return p
    return None


def _build_now_image_candidates(
    feature_id: object,
    section_id: object,
    now_year_used: object,
) -> List[str]:
    stem = f"tile_{feature_id}_{section_id}_now_{now_year_used}"
    exts = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ""]
    return [f"{stem}{ext}" for ext in exts]


def _resolve_now_image_path(row: pd.Series, images_dir: Path) -> Optional[Path]:
    candidates = _build_now_image_candidates(
        row["feature_id"],
        row["section_id"],
        int(float(row["now_year_used"])),
    )
    for name in candidates:
        p = images_dir / name
        if p.exists() and p.is_file():
            return p
    return None


def _compute_meta_features(df: pd.DataFrame) -> np.ndarray:
    points_by_year: Dict[int, List[Dict[str, object]]] = {}
    for idx, row in df.iterrows():
        year = int(row["past_year_used"])
        points_by_year.setdefault(year, []).append(
            {
                "id": int(idx),
                "lon": float(row["past_tree_lon"]),
                "lat": float(row["past_tree_lat"]),
            }
        )

    ndvi_map: Dict[int, float] = {}
    precip_map: Dict[int, float] = {}
    summer_map: Dict[int, float] = {}
    winter_map: Dict[int, float] = {}

    for year, points in points_by_year.items():
        ndvi_res = batch_average_ndvi_for_year(points, year=year)
        precip_res = batch_average_precip_last_year(points, year=year)
        temp_res = batch_average_temp_summer_winter(points, year=year)

        for r in ndvi_res:
            ndvi_map[int(r["id"])] = float(r["ndvi"]) if r["ndvi"] is not None else np.nan
        for r in precip_res:
            precip_map[int(r["id"])] = (
                float(r["precip"]) if r["precip"] is not None else np.nan
            )
        for r in temp_res:
            summer_map[int(r["id"])] = (
                float(r["summer_avg_c"]) if r["summer_avg_c"] is not None else np.nan
            )
            winter_map[int(r["id"])] = (
                float(r["winter_avg_c"]) if r["winter_avg_c"] is not None else np.nan
            )

    meta_rows = []
    for idx, row in df.iterrows():
        rid = int(idx)
        tree_num = float(row["past_ntrees"])
        meta_rows.append(
            [
                tree_num,
                ndvi_map.get(rid, np.nan),
                precip_map.get(rid, np.nan),
                summer_map.get(rid, np.nan),
                winter_map.get(rid, np.nan),
            ]
        )

    meta = np.array(meta_rows, dtype=np.float32)

    finite_mask = np.isfinite(meta)
    meta[~finite_mask] = np.nan

    col_means = np.nanmean(meta, axis=0)
    all_nan_cols = np.isnan(col_means)
    col_means[all_nan_cols] = 0.0

    missing_mask = np.isnan(meta)
    if missing_mask.any():
        meta[missing_mask] = np.take(col_means, np.where(missing_mask)[1])

    if not np.isfinite(meta).all():
        raise ValueError("Meta features still contain non-finite values after imputation.")

    return meta


def _compute_or_load_meta_features(df: pd.DataFrame, meta_cache_csv: Path) -> np.ndarray:
    key_cols = ["feature_id", "section_id", "past_year_used"]
    meta_cols = ["meta_tree_num", "meta_ndvi", "meta_precip", "meta_summer_c", "meta_winter_c"]
    cache_cols = key_cols + meta_cols

    work = df.copy()
    for c in key_cols:
        work[c] = work[c].astype(str)
    work["meta_tree_num"] = work["past_ntrees"].astype(np.float32)

    cached = pd.DataFrame(columns=cache_cols)
    if meta_cache_csv.exists():
        cached = pd.read_csv(meta_cache_csv)
        if not set(cache_cols).issubset(cached.columns):
            cached = pd.DataFrame(columns=cache_cols)
        else:
            cached = cached[cache_cols].copy()
            for c in key_cols:
                cached[c] = cached[c].astype(str)
        print(f"Loaded metadata cache rows: {len(cached)} from {meta_cache_csv}")

    merged = work.merge(cached, on=key_cols, how="left", suffixes=("", "_cached"))

    for c in meta_cols:
        merged[c] = pd.to_numeric(merged[c], errors="coerce")
    missing_meta_mask = merged[meta_cols].isna().any(axis=1)
    missing_count = int(missing_meta_mask.sum())

    if missing_count:
        print(f"Metadata cache misses: {missing_count}; querying APIs for missing rows.")
        missing_df = df.loc[missing_meta_mask.values].reset_index(drop=True)
        computed = _compute_meta_features(missing_df)
        merged.loc[missing_meta_mask, "meta_tree_num"] = computed[:, 0]
        merged.loc[missing_meta_mask, "meta_ndvi"] = computed[:, 1]
        merged.loc[missing_meta_mask, "meta_precip"] = computed[:, 2]
        merged.loc[missing_meta_mask, "meta_summer_c"] = computed[:, 3]
        merged.loc[missing_meta_mask, "meta_winter_c"] = computed[:, 4]
    else:
        print("Metadata cache hit for all rows; no API calls needed.")

    meta = merged[meta_cols].to_numpy(dtype=np.float32)
    if not np.isfinite(meta).all():
        raise ValueError("Meta cache contains non-finite values after refresh.")

    refreshed_cache = merged[cache_cols].copy()
    combined_cache = pd.concat([cached, refreshed_cache], ignore_index=True)
    combined_cache = combined_cache.drop_duplicates(subset=key_cols, keep="last")
    meta_cache_csv.parent.mkdir(parents=True, exist_ok=True)
    combined_cache.to_csv(meta_cache_csv, index=False)
    print(f"Saved metadata cache rows: {len(combined_cache)} to {meta_cache_csv}")

    return meta


def _load_training_frame(
    csv_path: Path,
    images_dir: Path,
    meta_cache_csv: Path,
) -> Tuple[pd.DataFrame, List[Path], np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)

    required_cols = [
        "feature_id",
        "section_id",
        "past_year_used",
        "past_tree_lon",
        "past_tree_lat",
        "past_ntrees",
        "delta_distance_m",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {missing}")

    df = df.copy().reset_index(drop=True)

    # Coerce numeric training fields so malformed values are treated as missing.
    numeric_cols = [
        "past_year_used",
        "past_tree_lon",
        "past_tree_lat",
        "past_ntrees",
        "delta_distance_m",
    ]
    if "now_year_used" in df.columns:
        numeric_cols.append("now_year_used")
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows that are missing required training values.
    missing_required_mask = df[required_cols].isna().any(axis=1)
    dropped_missing_required = int(missing_required_mask.sum())
    if dropped_missing_required:
        print(
            f"Warning: {dropped_missing_required} rows dropped because required values were missing."
        )
    df = df.loc[~missing_required_mask].reset_index(drop=True)
    if df.empty:
        raise ValueError("No rows remain after filtering missing required training values.")

    # Drop rows with negative targets.
    non_negative_target_mask = df["delta_distance_m"].ge(0)
    dropped_negative_targets = int((~non_negative_target_mask).sum())
    if dropped_negative_targets:
        print(
            f"Warning: {dropped_negative_targets} rows dropped because delta_distance_m was negative."
        )
    df = df.loc[non_negative_target_mask].reset_index(drop=True)
    if df.empty:
        raise ValueError(
            "No rows remain after filtering delta_distance_m. All targets were negative."
        )

    image_paths: List[Path] = []
    keep_rows: List[int] = []

    for i, row in df.iterrows():
        resolved = _resolve_image_path(row, images_dir)
        if resolved is not None:
            image_paths.append(resolved)
            keep_rows.append(i)

    if not keep_rows:
        raise ValueError(
            f"No training rows had a matching image in {images_dir}. "
            "Expected names like tile_{feature_id}_{section_id}_past_{past_year_used}.png"
        )

    if len(keep_rows) < len(df):
        print(
            f"Warning: {len(df) - len(keep_rows)} rows dropped because image files were not found."
        )

    df = df.iloc[keep_rows].reset_index(drop=True)

    meta = _compute_or_load_meta_features(df, meta_cache_csv=meta_cache_csv)
    targets = df["delta_distance_m"].astype(np.float32).to_numpy()

    return df, image_paths, meta, targets


def _make_collate_fn(processor: AutoImageProcessor, device: torch.device):
    def collate(batch: Sequence[Dict[str, object]]) -> Dict[str, torch.Tensor]:
        images = [item["image"] for item in batch]
        meta = torch.stack([item["meta"] for item in batch]).to(device)
        target = torch.stack([item["target"] for item in batch]).to(device)

        inputs = processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)

        return {
            "pixel_values": pixel_values,
            "meta": meta,
            "target": target,
        }

    return collate


def train(
    csv_path: str,
    images_dir: str,
    save_path: str = "image_meta_regressor.pt",
    meta_cache_csv: Optional[str] = None,
    epochs: int = 10,
    batch_size: int = 8,
    lr: float = 1e-4,
    train_ratio: float = 0.8,
    seed: int = 42,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    csv_file = Path(csv_path)
    img_dir = Path(images_dir)
    cache_file = (
        Path(meta_cache_csv)
        if meta_cache_csv is not None
        else csv_file.with_name(f"{csv_file.stem}_meta_cache.csv")
    )

    _, image_paths, meta, targets = _load_training_frame(
        csv_file,
        img_dir,
        meta_cache_csv=cache_file,
    )

    if not np.isfinite(targets).all():
        raise ValueError("Targets contain non-finite values after filtering.")

    meta_mean = meta.mean(axis=0, keepdims=True)
    meta_std = meta.std(axis=0, keepdims=True)
    meta_std[meta_std == 0] = 1.0
    meta_norm = (meta - meta_mean) / meta_std
    if not np.isfinite(meta_norm).all():
        raise ValueError("Normalized meta features contain non-finite values.")

    dataset = TreelineDataset(image_paths=image_paths, meta_features=meta_norm, targets=targets)

    if len(dataset) < 2:
        raise ValueError("Need at least 2 samples after filtering to do an 80/20 split.")

    train_len = int(len(dataset) * train_ratio)
    train_len = max(1, min(train_len, len(dataset) - 1))
    val_len = len(dataset) - train_len

    g = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(dataset, [train_len, val_len], generator=g)

    processor = AutoImageProcessor.from_pretrained("apple/mobilevit-small")
    collate_fn = _make_collate_fn(processor=processor, device=device)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    model = ImageMetaRegressor(meta_dim=meta.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        train_losses = []
        skipped_train_batches = 0

        for batch in train_loader:
            optimizer.zero_grad()
            preds = model(pixel_values=batch["pixel_values"], meta_features=batch["meta"])
            loss = criterion(preds, batch["target"])
            if not torch.isfinite(loss):
                skipped_train_batches += 1
                continue
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        val_abs = []
        skipped_val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                preds = model(pixel_values=batch["pixel_values"], meta_features=batch["meta"])
                loss = criterion(preds, batch["target"])
                if not torch.isfinite(loss):
                    skipped_val_batches += 1
                    continue
                val_losses.append(loss.item())
                val_abs.append(torch.mean(torch.abs(preds - batch["target"])).item())

        if not train_losses:
            raise RuntimeError(
                f"All train batches were non-finite in epoch {epoch + 1}. "
                "Check image files and meta feature generation."
            )
        if not val_losses:
            raise RuntimeError(
                f"All validation batches were non-finite in epoch {epoch + 1}. "
                "Check image files and meta feature generation."
            )

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        val_mae = float(np.mean(val_abs))

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"train_mse={train_loss:.4f} val_mse={val_loss:.4f} val_mae={val_mae:.4f} "
            f"(skipped_train_batches={skipped_train_batches}, skipped_val_batches={skipped_val_batches})"
        )

    save_file = Path(save_path)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "meta_dim": int(meta.shape[1]),
            "meta_mean": meta_mean.astype(np.float32),
            "meta_std": meta_std.astype(np.float32),
            "train_ratio": train_ratio,
            "seed": seed,
        },
        save_file,
    )

    print(f"Model saved to {save_file.resolve()}")


def _load_checkpoint(checkpoint_path: Path, device: torch.device):
    # PyTorch 2.6 defaults to weights_only=True, which can fail when the checkpoint
    # includes numpy metadata. For local trusted checkpoints, load with weights_only=False.
    try:
        return torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location=device)


def inference(
    model_path: str,
    image_path: str,
    meta_features: Sequence[float],
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Inference on: {device}")

    checkpoint = _load_checkpoint(Path(model_path), device)
    meta_dim = int(checkpoint["meta_dim"])

    if len(meta_features) != meta_dim:
        raise ValueError(f"Expected {meta_dim} meta features, got {len(meta_features)}")

    processor = AutoImageProcessor.from_pretrained("apple/mobilevit-small")
    model = ImageMetaRegressor(meta_dim=meta_dim).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=[image], return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    meta = np.array(meta_features, dtype=np.float32).reshape(1, -1)
    meta_mean = checkpoint.get("meta_mean")
    meta_std = checkpoint.get("meta_std")
    if meta_mean is not None and meta_std is not None:
        meta = (meta - meta_mean) / np.where(meta_std == 0, 1.0, meta_std)

    meta_tensor = torch.tensor(meta, dtype=torch.float32, device=device)

    with torch.no_grad():
        pred = model(pixel_values=pixel_values, meta_features=meta_tensor)

    value = float(pred.cpu().numpy()[0])
    print(f"Prediction: {value}")
    return value


def inference_from_cache(
    model_path: str,
    csv_path: str,
    images_dir: str,
    meta_cache_csv: str,
    output_csv: str,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Inference on: {device}")

    model_file = Path(model_path)
    csv_file = Path(csv_path)
    img_dir = Path(images_dir)
    cache_file = Path(meta_cache_csv)
    out_file = Path(output_csv)

    checkpoint = _load_checkpoint(model_file, device)
    meta_dim = int(checkpoint["meta_dim"])
    if meta_dim != 5:
        raise ValueError(
            f"Unsupported checkpoint meta_dim={meta_dim}. "
            "Current training expects 5 metadata features."
        )

    processor = AutoImageProcessor.from_pretrained("apple/mobilevit-small")
    model = ImageMetaRegressor(meta_dim=meta_dim).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    source = pd.read_csv(csv_file)
    cache = pd.read_csv(cache_file)

    key_cols = ["feature_id", "section_id", "past_year_used"]
    required_cache_cols = key_cols + [
        "meta_tree_num",
        "meta_ndvi",
        "meta_precip",
        "meta_summer_c",
        "meta_winter_c",
    ]
    missing_cache = [c for c in required_cache_cols if c not in cache.columns]
    if missing_cache:
        raise ValueError(f"Missing required cache columns in {cache_file}: {missing_cache}")

    if {"center_lon", "center_lat"}.issubset(source.columns):
        coord_lon_col = "center_lon"
        coord_lat_col = "center_lat"
    elif {"past_tree_lon", "past_tree_lat"}.issubset(source.columns):
        coord_lon_col = "past_tree_lon"
        coord_lat_col = "past_tree_lat"
    else:
        raise ValueError(
            f"Source CSV must contain coordinates via center_lon/center_lat "
            f"or past_tree_lon/past_tree_lat: {csv_file}"
        )

    required_source_cols = key_cols + [
        "now_year_used",
        "delta_distance_m",
        "now_distance_m",
        coord_lon_col,
        coord_lat_col,
    ]
    missing_source = [c for c in required_source_cols if c not in source.columns]
    if missing_source:
        raise ValueError(f"Missing required source columns in {csv_file}: {missing_source}")

    source = source.copy()
    cache = cache.copy()
    for c in key_cols:
        source[c] = source[c].astype(str)
        cache[c] = cache[c].astype(str)

    source = source.drop_duplicates(subset=key_cols, keep="first")
    merged = cache.merge(source[required_source_cols], on=key_cols, how="left")

    numeric_cols = [
        "now_year_used",
        "delta_distance_m",
        "now_distance_m",
        coord_lon_col,
        coord_lat_col,
        "meta_tree_num",
        "meta_ndvi",
        "meta_precip",
        "meta_summer_c",
        "meta_winter_c",
    ]
    for col in numeric_cols:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")

    valid_mask = merged[numeric_cols].notna().all(axis=1)
    merged = merged.loc[valid_mask].reset_index(drop=True)
    if merged.empty:
        raise ValueError("No valid rows available after joining cache with source CSV.")

    image_paths: List[Path] = []
    keep_rows: List[int] = []
    for i, row in merged.iterrows():
        img_path = _resolve_now_image_path(row, img_dir)
        if img_path is not None:
            image_paths.append(img_path)
            keep_rows.append(i)

    if not keep_rows:
        raise ValueError(
            f"No valid rows had matching now-images in {img_dir}. "
            "Expected names like tile_{feature_id}_{section_id}_now_{now_year_used}.png"
        )

    if len(keep_rows) < len(merged):
        print(
            f"Warning: {len(merged) - len(keep_rows)} rows dropped because now-images were not found."
        )

    merged = merged.iloc[keep_rows].reset_index(drop=True)
    meta = merged[
        ["meta_tree_num", "meta_ndvi", "meta_precip", "meta_summer_c", "meta_winter_c"]
    ].to_numpy(dtype=np.float32)

    meta_mean = checkpoint.get("meta_mean")
    meta_std = checkpoint.get("meta_std")
    if meta_mean is not None and meta_std is not None:
        meta = (meta - meta_mean) / np.where(meta_std == 0, 1.0, meta_std)

    if not np.isfinite(meta).all():
        raise ValueError("Inference metadata contains non-finite values.")

    preds: List[float] = []
    for i in range(len(merged)):
        image = Image.open(image_paths[i]).convert("RGB")
        inputs = processor(images=[image], return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
        meta_tensor = torch.tensor(meta[i : i + 1], dtype=torch.float32, device=device)
        with torch.no_grad():
            pred = model(pixel_values=pixel_values, meta_features=meta_tensor)
        preds.append(float(pred.cpu().numpy()[0]))

    output = pd.DataFrame(
        {
            "feature_id": merged["feature_id"],
            "section_id": merged["section_id"],
            "coord_lon": merged[coord_lon_col].astype(np.float32),
            "coord_lat": merged[coord_lat_col].astype(np.float32),
            "now_distance_m": merged["now_distance_m"].astype(np.float32),
            "predicted_future_delta_distance_m": np.array(preds, dtype=np.float32),
            "image_path": [str(p) for p in image_paths],
        }
    )

    out_file.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(out_file, index=False)
    print(f"Saved inference predictions: {len(output)} rows to {out_file.resolve()}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ViT + metadata regressor for treeline distance.")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "infer"])
    parser.add_argument("--csv", type=str, required=True, help="Training CSV path")
    parser.add_argument("--images-dir", type=str, default="ViT/images", help="Directory with tiles")
    parser.add_argument("--save-path", type=str, default="image_meta_regressor.pt")
    parser.add_argument(
        "--meta-cache-csv",
        type=str,
        default=None,
        help="Optional metadata cache CSV path. Defaults to <csv_stem>_meta_cache.csv next to training CSV.",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-path", type=str, default=None, help="Checkpoint path for inference mode.")
    parser.add_argument(
        "--inference-output-csv",
        type=str,
        default=None,
        help="Output CSV for inference results. Defaults to <cache_stem>_predictions.csv.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.mode == "train":
        ee.Authenticate()
        ee.Initialize(project='caramel-park-488923-j2')
        train(
            csv_path=args.csv,
            images_dir=args.images_dir,
            save_path=args.save_path,
            meta_cache_csv=args.meta_cache_csv,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            train_ratio=args.train_ratio,
            seed=args.seed,
        )
    else:
        cache_file = (
            Path(args.meta_cache_csv)
            if args.meta_cache_csv is not None
            else Path(args.csv).with_name(f"{Path(args.csv).stem}_meta_cache.csv")
        )
        model_file = args.model_path if args.model_path is not None else args.save_path
        output_csv = (
            args.inference_output_csv
            if args.inference_output_csv is not None
            else str(cache_file.with_name(f"{cache_file.stem}_predictions.csv"))
        )
        inference_from_cache(
            model_path=model_file,
            csv_path=args.csv,
            images_dir=args.images_dir,
            meta_cache_csv=str(cache_file),
            output_csv=output_csv,
        )
