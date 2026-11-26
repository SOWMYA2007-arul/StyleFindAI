# Segmentation benchmark (synthetic)

Generated 2026-11-22 13:00 UTC by `scripts/evaluate_segmentation.py`.
Data: `dataset/evaluation/segmentation_synthetic` (260 images). Settings: {'segmentation_strategy': 'auto', 'garment_detector': 'none', 'person_segmenter': 'yolo', 'human_parser': 'none', 'enable_sam2': False, 'pose_estimator': 'none', 'boundary_refinement': 'guided', 'min_segmentation_confidence': 0.4, 'ai_device': 'cpu'}

> **SYNTHETIC drawings** with exact masks: regression evidence only, NOT real-world accuracy.

## Overall

| method | IoU | Dice | precision | recall | boundary F1 | BF1 (2 px) | inst. IoU | AP50 | mAP | fallback | p50 ms | p95 ms | CPU RSS peak MB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| legacy_v1 | 0.781 | 0.863 | 0.790 | 0.978 | 0.587 | 0.546 | 0.211 | 0.000 | 0.000 | n/a | 4 | 5 | 680.5 |
| foreground | 0.879 | 0.921 | 0.887 | 0.969 | 0.833 | 0.734 | 0.236 | 0.005 | 0.002 | n/a | 10 | 13 | 683.9 |
| foreground_guided | 0.889 | 0.926 | 0.896 | 0.970 | 0.847 | 0.750 | 0.238 | 0.005 | 0.002 | n/a | 12 | 16 | 690.2 |
| pipeline_light | 0.919 | 0.943 | 0.927 | 0.973 | 0.881 | 0.862 | 0.926 | 0.906 | 0.817 | 1% | 18 | 122 | 703.1 |
| pipeline_auto | 0.919 | 0.943 | 0.927 | 0.973 | 0.881 | 0.862 | 0.926 | 0.906 | 0.817 | 1% | 19 | 203 | 963.9 |

## IoU by image category

| category | legacy_v1 | foreground | foreground_guided | pipeline_light | pipeline_auto |
|---|---|---|---|---|---|
| clutter | 0.907 | 0.945 | 0.959 | 0.963 | 0.963 |
| complex_bg | 0.510 | 0.600 | 0.602 | 0.627 | 0.627 |
| dark_on_dark | 0.619 | 0.990 | 0.996 | 0.996 | 0.996 |
| gradient_bg | 0.685 | 0.964 | 0.973 | 0.973 | 0.973 |
| multi_garment | 0.628 | 0.661 | 0.667 | 0.733 | 0.733 |
| occluded | 0.835 | 0.833 | 0.845 | 0.973 | 0.973 |
| patterned | 0.981 | 0.980 | 0.993 | 0.993 | 0.993 |
| person_wearing | 0.658 | 0.691 | 0.698 | 0.791 | 0.791 |
| product_only | 0.978 | 0.976 | 0.990 | 0.990 | 0.990 |
| shadow | 0.815 | 0.972 | 0.985 | 0.985 | 0.985 |
| thin_details | 0.953 | 0.950 | 0.962 | 0.962 | 0.962 |
| transparent | 0.974 | 0.972 | 0.985 | 0.985 | 0.985 |
| white_on_white | 0.606 | 0.897 | 0.901 | 0.975 | 0.975 |

## boundary F1 by image category

| category | legacy_v1 | foreground | foreground_guided | pipeline_light | pipeline_auto |
|---|---|---|---|---|---|
| clutter | 0.861 | 0.940 | 0.941 | 0.944 | 0.944 |
| complex_bg | 0.338 | 0.518 | 0.519 | 0.564 | 0.564 |
| dark_on_dark | 0.137 | 0.999 | 0.997 | 0.997 | 0.997 |
| gradient_bg | 0.294 | 0.985 | 0.984 | 0.985 | 0.985 |
| multi_garment | 0.367 | 0.565 | 0.567 | 0.635 | 0.635 |
| occluded | 0.878 | 0.877 | 0.878 | 0.984 | 0.984 |
| patterned | 0.998 | 0.997 | 0.997 | 0.997 | 0.997 |
| person_wearing | 0.456 | 0.539 | 0.539 | 0.686 | 0.686 |
| product_only | 0.998 | 0.998 | 0.998 | 0.998 | 0.998 |
| shadow | 0.500 | 0.998 | 0.998 | 0.998 | 0.998 |
| thin_details | 0.696 | 0.699 | 0.708 | 0.708 | 0.708 |
| transparent | 0.972 | 0.970 | 0.971 | 0.971 | 0.971 |
| white_on_white | 0.139 | 0.742 | 0.915 | 0.987 | 0.987 |

Notes: a pipeline fallback is scored as the full image. boundary F1 tolerance = 0.8 % of the image diagonal;
BF1 (2 px) is the strict version that rewards thin details (straps, tassels, lace edges).
