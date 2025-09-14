>> python panorama_pipeline.py --mode full --input_dir calibration_run_min --triplet_index 1
================================================================================
STARTING FULL PANORAMA PIPELINE
================================================================================
============================================================
STEP 1: GENERATING PANORAMA
============================================================
Loading images from: calibration_run_min
Found 9 images
Resizing images...
Original Size: (800, 600) -> 480,000 px
Medium Size: (800, 600) -> 480,000 px
Low Size: (365, 274) -> 100,010 px
Final Size: (800, 600) -> 480,000 px
Detecting features...
Features detected per image: [2236, 978, 1766, 2125, 3380, 3598, 2833, 1928, 3196]
Matching features...
Feature matching confidence matrix:
  [[0.         0.68345324 0.52356021 0.35714286 0.38167939 0.33613445
    0.34482759 0.34246575 0.41860465]
  [0.68345324 0.         0.99750623 0.41176471 0.34482759 0.34482759
    0.36585366 0.26845638 0.26315789]
  [0.52356021 0.99750623 0.         0.48648649 0.4        0.40268456
    0.2919708  0.53435115 0.05494505]
  [0.35714286 0.41176471 0.48648649 0.         1.17073171 1.38381201
    0.20304569 0.11976048 0.38709677]
  [0.38167939 0.34482759 0.4        1.17073171 0.         2.28412256
    1.89097104 0.32       0.35714286]
  [0.33613445 0.34482759 0.40268456 1.38381201 2.28412256 0.
    1.43338954 0.63291139 0.27972028]
  [0.34482759 0.36585366 0.2919708  0.20304569 1.89097104 1.43338954
    0.         1.22850123 0.94339623]
  [0.34246575 0.26845638 0.53435115 0.11976048 0.32       0.63291139
    1.22850123 0.         1.06639839]
  [0.41860465 0.26315789 0.05494505 0.38709677 0.35714286 0.27972028
    0.94339623 1.06639839 0.        ]]
Using confidence threshold: 0.419
Relevant matches found:
  Matched Image 1 to Image 2
  Matched Image 1 to Image 9
  Matched Image 2 to Image 3
  Matched Image 3 to Image 4
  Matched Image 4 to Image 5
  Matched Image 5 to Image 6
  Matched Image 6 to Image 7
  Matched Image 7 to Image 8
  Matched Image 8 to Image 9
Estimating camera parameters...
Estimated 9 camera parameters
Setting up warper...
Full calibration saved: calibration_params/calibration_20250803_230240.npz
Warping low resolution images...
Warping final resolution images...
Finding optimal seams...
Applying exposure compensation...
Blending images...
Panorama saved: pipeline_outputs/pipeline_panorama_1754242360.539389.png
STEP 1 COMPLETE
============================================================
STEP 2: EXTRACTING TRIPLET CALIBRATION
============================================================
esp_1: selected 7_esp_1_frame_1753211237.912193.jpg (2/3)
esp_2: selected 4_esp_2_frame_1753211237.941754.jpg (2/3)
esp_3: selected 1_esp_3_frame_1753211238.719612.jpg (2/3)
Selected triplet images (index 1):
  Camera 1: 7_esp_1_frame_1753211237.912193.jpg
  Camera 2: 4_esp_2_frame_1753211237.941754.jpg
  Camera 3: 1_esp_3_frame_1753211238.719612.jpg
Using reference panorama: pipeline_panorama_1754242360.539389.png
Panorama resized to: (600, 2082)
Triplet images at: [(600, 800), (600, 800), (600, 800)]
Detecting features...
Features found - Triplet: [1928, 3380, 978]
Features found - Panorama: 4854
Matching features...
Confidence matrix:
  [[0.         0.29850746 0.         1.43727162]
  [0.29850746 0.         0.29850746 2.05479452]
  [0.         0.29850746 0.         2.33021077]
  [1.43727162 2.05479452 2.33021077 0.        ]]
Estimating camera parameters...
Triplet calibration saved:
  - NumPy: triplet_calibration/triplet_calibration_20250803_230240.npz
  - JSON: triplet_calibration/triplet_calibration_20250803_230240.json
STEP 2 COMPLETE
============================================================
STEP 3: STITCHING TRIPLET IMAGES
============================================================
Using calibration: triplet_calibration_20250803_230240.npz
Stitching images:
  Image 1: 7_esp_1_frame_1753211237.912193.jpg
  Image 2: 4_esp_2_frame_1753211237.941754.jpg
  Image 3: 1_esp_3_frame_1753211238.719612.jpg
Loading triplet calibration from: triplet_calibration_20250803_230240.npz
Loaded calibration: 3 cameras, scale: 1746.634487030614
Processing images...
Warping low resolution images...
Warping final resolution images...
Finding optimal seams...
Applying exposure compensation...
Blending images...
Panorama saved to: pipeline_outputs/triplet_stitched_1754242360.817411.png
Triplet panorama saved: pipeline_outputs/triplet_stitched_1754242360.817411.png
STEP 3 COMPLETE
================================================================================
PIPELINE COMPLETE!
================================================================================
Duration: 0:00:01.342417
Panorama: pipeline_outputs/pipeline_panorama_1754242360.539389.png
Calibration: triplet_calibration/triplet_calibration_20250803_230240.npz
Test stitch: pipeline_outputs/triplet_stitched_1754242360.817411.png
================================================================================

============================================================
PIPELINE RESULTS SUMMARY
============================================================
Input Directory: calibration_run_min
Triplet Index: 1
Duration: 0:00:01.342417

Generated Files:
  Panorama: pipeline_outputs/pipeline_panorama_1754242360.539389.png
  Calibration: triplet_calibration/triplet_calibration_20250803_230240.npz
  Test Stitch: pipeline_outputs/triplet_stitched_1754242360.817411.png

Triplet Images Used:
  Camera 1: 7_esp_1_frame_1753211237.912193.jpg
  Camera 2: 4_esp_2_frame_1753211237.941754.jpg
  Camera 3: 1_esp_3_frame_1753211238.719612.jpg
============================================================