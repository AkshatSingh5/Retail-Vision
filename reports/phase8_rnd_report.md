# Retail Vision — Phase 8 R&D Report

Generated: 2026-08-21T09:11:47.011131+00:00

## Verdict

**YOLO26m alone is not sufficient for real-world retail checkout.**

**The system is not production-ready** based on measured results.

A two-stage detector + recognizer is recommended *after* a much larger real-photo dataset exists. A second embedding model will not fix misses: at the live threshold the detector often produces no box at all.

### Why this is not production-ready

- Held-out test recall is 0.375 and mAP50 is 0.463 on only 9 images.
- At the live 0.50 confidence threshold, identity benchmark recall was 0.0.
- Robustness over 1080 synthetic variations of those 9 photos was 0.65% hits.
- The dataset does not contain 1000 unique real captures per SKU.
- Inference is CPU-only on this machine; CUDA PyTorch is not installed.
- Maggi was confused with Kurkure on the test set.

## Measured metrics

| Metric | Value | Source |
| --- | --- | --- |
| Detection accuracy (mAP50) | 0.4625 | Ultralytics test split |
| Recognition accuracy (live 0.50) | 0.0 | identity benchmark |
| Precision | 0.6443 | test split |
| Recall | 0.375 | test split |
| mAP50 | 0.4625 | test split |
| mAP50-95 | 0.2205 | test split |
| False positive rate | 0 background FPs in Phase 4 confusion; live-threshold misses dominate instead | confusion report |
| False negative rate | 0.625 | 1 − recall |
| FPS (condition sweep) | 7.01 | mean 142.6 ms |
| Inference latency | 142.6 ms | CPU YOLO26m |
| Average transaction processing time | 18.2 ms | checkout + PDF |
| Named-condition hit rate | 0.0 | 171 probes at conf 0.50 |
| 1080-variation hit rate | 0.0065 | Phase 4 protocol |

## Architecture

```text
Camera
  -> YOLO26m detection
  -> Multi-object tracking (ByteTrack / IoU)
  -> Product identity (class_id)
  -> Database SKU / price / tax_rate
  -> Acceptance gate (confidence, catalog, valid price)
  -> Cart quantity from unique track_id
  -> Tax from database
  -> Bill + PDF invoice
```

Prices are never invented by the vision model. Uncertain detections are rejected with a manual-verify message instead of being added to the bill.

## Dataset methodology

- 8 SKUs in `products/registry.yaml`
- 74 labeled images, session-aware 70/20/10 split (48/17/9)
- Seed photos from Open Food Facts / Wikimedia, not store-camera captures
- Robustness set: held-out test photos probed under named transforms (angle, rotation, distance, lighting, occlusion, background, scale, orientation)
- **Augmented copies are not counted as independent real-world evidence.**

## Model and training

- Model: YOLO26m (`ultralytics`), weights `retail_yolo26m_v2.pt`
- Train imgsz: 416
- Epochs: 20, batch: 8, freeze: 10
- Device during training: cpu
- Mosaic/mixup: off in v2

## Accuracy (held-out test set)

- Precision: 0.6443
- Recall: 0.375
- F1: 0.4741
- mAP50: 0.4625
- mAP50-95: 0.2205
- False-negative rate (1 - recall): 0.625
- Background false positives in the Phase 4 confusion report: 0 per class

Live operating point (`CONFIDENCE_THRESHOLD=0.50`): identity benchmark detection recall **0.0** on the 9-image test split.

## Recognition performance

- Prior 1000+ variation protocol: 1080 evaluations, hit rate **0.0065** (7/1080).
- Phase 8 named-condition sweep: 171 evaluations, hit rate **0.0**.
- Condition sweep uses geometric/photometric probes of the same 9 test photos. It is a robustness diagnostic, not a claim of 1000 real SKUs.

### Similar products

- Coke vs Pepsi / Sprite: no Coke-Pepsi swaps on the tiny test set; several classes simply miss.
- Maggi Noodles -> Kurkure Masala Munch: 1 confusion in Phase 4.
- Different Lays flavours / Maggi variants / pack sizes: **not in the catalog**, so they were not measured.

- Identity among low-confidence matches: 0.3333

## Tracking performance

- One Coke across 100 frames: quantity 1 (PASS)
- Two physical Cokes: quantity 2 (PASS)
- Cart add requires `STABLE_FRAMES` consecutive hits and the acceptance gate

## Billing accuracy

- class_id 0 -> SKU COKE500 -> price 40 from SQLite (PASS)
- Five-product cart (Coke x2 + Lays + Maggi + Pepsi + KitKat): subtotal 184, tax 33.12, total 217.12
- Average checkout / PDF time: 18.2 ms
- Low-confidence tracks are not billed (PASS)

## FPS and latency

- Device: cpu (installed PyTorch is CPU-only on this machine)
- Condition-sweep mean latency: 142.6 ms
- Approx FPS from that latency: 7.01

### Stress (composite scenes)

| Objects | Detections | Latency ms | FPS | CPU % | Memory MB | GPU |
| --- | --- | --- | --- | --- | --- | --- |
| 5 | 0 | 95.5 | 10.47 | 705.6 | 471.4 | n/a |
| 10 | 0 | 115.7 | 8.64 | 698.1 | 476.5 | n/a |
| 20 | 0 | 176.6 | 5.66 | 806.5 | 470.2 | n/a |
| 30 | 0 | 204.1 | 4.9 | 596.6 | 475.6 | n/a |

### Optimization investigated (not enabled by default)

- Image size 320/416/640 latency probe (do not drop imgsz solely for FPS)
- `FRAME_SKIP` in `.env` (default 0)
- `INFER_IMGSZ` optional override
- GPU acceleration: blocked here by a CPU PyTorch wheel and an old NVIDIA driver (CUDA 11.1)
- Export/quantization: not deployed; accuracy is already below retail bar at FP32

Latency vs image size (speed probe only; accuracy was not re-scored):

| imgsz | Mean latency ms |
| --- | --- |
| 320 | 118.4 |
| 416 | 161.4 |
| 640 | 408.2 |

CPU % in the stress table is a short `psutil` sample of a multi-thread CPU inference process and can exceed 100. It is not a calibrated hardware meter. GPU usage is n/a because this PyTorch build has no CUDA.

Named-condition hit rates (all at live confidence 0.50):

| Category | Hits | Total | Rate |
| --- | --- | --- | --- |
| Angle | 0 | 36 | 0.0 |
| Background | 0 | 9 | 0.0 |
| Distance | 0 | 18 | 0.0 |
| Lighting | 0 | 36 | 0.0 |
| Occlusion | 0 | 9 | 0.0 |
| Orientation | 0 | 18 | 0.0 |
| Rotation | 0 | 36 | 0.0 |
| Scale | 0 | 9 | 0.0 |

| Condition | Hits | Total | Rate |
| --- | --- | --- | --- |
| front_view | 0 | 9 | 0.0 |
| back_view_proxy_flip | 0 | 9 | 0.0 |
| left_view_proxy | 0 | 9 | 0.0 |
| right_view_proxy | 0 | 9 | 0.0 |
| rotation_15 | 0 | 9 | 0.0 |
| rotation_30 | 0 | 9 | 0.0 |
| rotation_45 | 0 | 9 | 0.0 |
| rotation_90 | 0 | 9 | 0.0 |
| upside_down | 0 | 9 | 0.0 |
| tilted | 0 | 9 | 0.0 |
| close_distance | 0 | 9 | 0.0 |
| far_distance | 0 | 9 | 0.0 |
| scale_small | 0 | 9 | 0.0 |
| lighting | 0 | 9 | 0.0 |
| low_light | 0 | 9 | 0.0 |
| shadows | 0 | 9 | 0.0 |
| reflections | 0 | 9 | 0.0 |
| partial_occlusion | 0 | 9 | 0.0 |
| cluttered_background | 0 | 9 | 0.0 |

## Failure cases handled

- Unknown product / not in database: rejected, operator message
- Low confidence: rejected, operator message
- Invalid price: rejected, operator message
- Camera disconnect: POS placeholder, no silent cart adds
- Database unavailable: camera loop continues, cart update skipped
- Model missing: POS shows model unavailable
- Duplicate tracks: unique `track_id` quantity

## Hardware requirements (this machine vs retail target)

| Item | Measured | Retail target |
| --- | --- | --- |
| OS | Windows 11 | Windows/Linux POS PC |
| GPU | GTX 1650, driver 457.49 / CUDA 11.1, unused | CUDA GPU with current driver |
| PyTorch | 2.13.0+cpu | CUDA build matching the driver |
| CPU latency | ~150-300 ms/frame | <50 ms typical checkout |

## Limitations

- Eight SKUs and 74 images cannot represent a store catalog
- No dedicated front/back/left/right camera captures; some angle tests are proxies (flip/rotate)
- Similar flavour/size variants are not registered
- CPU inference is too slow for a snappy checkout lane
- Detector recall at the live threshold is the blocking failure, not cart math

## Future improvements

1. Capture 1000+ real in-store views per SKU (true angle, lighting, occlusion, clutter)
2. Retrain YOLO26m until live-threshold recall is high on a held-out camera set
3. Then evaluate a second-stage embedding model for remaining similar-SKU collisions
4. Install a CUDA PyTorch build after updating the NVIDIA driver
5. Add pack-size and flavour SKUs before claiming multi-variant recognition

## Suite result: 14 passed, 0 failed

