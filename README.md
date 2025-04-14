# Football Computer Vision Analysis

A computer vision project that analyzes football/soccer videos to track players, the ball, and provide detailed game analytics.

## Demo

![Football Analysis Demo](demo/output_demo.gif)

Sample output showing:
- Player detection and tracking
- Ball tracking
- Team classification
- Speed and distance measurements
- Camera movement visualization
- Ball possession tracking

## Features

- **Player Tracking**: Automated tracking of all players on the field using YOLOv8
- **Ball Tracking**: Continuous ball position tracking with interpolation
- **Team Assignment**: Automatic team classification based on jersey colors
- **Ball Possession**: Detection of which player/team has possession of the ball
- **Speed & Distance**: Real-time calculation of player speeds and distances covered
  - Smoothed speed calculations using 5-frame window
  - Speed validation (0.1-40 km/h range)
  - Distance tracking in real-world meters
- **Camera Movement Tracking**: Compensation for camera movement during analysis
- **Perspective Transform**: Convert camera view to top-down perspective
  - Hardcoded field dimensions (23.32m x 68m)
  - Automatic field bounds validation

## Requirements

### Python Dependencies
- Python 3.7+
- OpenCV
- NumPy
- YOLOv8
- Additional dependencies listed in `pyproject.toml`

### System Dependencies
- FFmpeg (for video processing)
  - macOS: `brew install ffmpeg`
  - Ubuntu/Debian: `sudo apt-get install ffmpeg`
  - Windows: Download from [FFmpeg website](https://ffmpeg.org/download.html)

## Setup

1. Clone the repository:
```bash
git clone [repository-url]
cd football_cv_project
```

2. Install Python dependencies using Poetry:
```bash
poetry install
```

3. Install FFmpeg (if not already installed):
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get install ffmpeg

# Windows
# Download and install from https://ffmpeg.org/download.html
```

4. Download the required model:
- Place the YOLOv8 model file (`yolov8x.pt`) in the `models/` directory

## Usage

1. Place your input video in the `input_videos/` directory

2. Incase, you want to fine tune on your own,visit the [Roboflow YOLOv8 Model Zoo](https://roboflow.com/)
   - Download the data using the roboflow using copy code option
   - Train(Finetune) the YOLOv8 model using roboflow data for around 100 epochs
   - Download the model checkpoint `yolo8x.pt`

3. Run the analysis:
```bash
python main.py
```

4. The system will:
   - Process the video using hardcoded field calibration
   - Track players and ball
   - Calculate speeds and distances
   - Generate visualizations
   - Save results in `output_videos/` and `calibration_results/`

## Output

The processed video includes visual annotations showing:
- Player tracking boxes with team colors
- Ball tracking
- Speed and distance information
- Ball possession indicators
- Camera movement visualization

Additional outputs in `calibration_results/`:
- Field transformation visualizations
- Player trajectories
- Heatmaps
- Field zones

## Field Calibration

The system uses hardcoded field calibration for accurate measurements:

1. **Field Dimensions**:
   - Length: 23.32 meters (visible portion)
   - Width: 68 meters (standard football field width)
   - These dimensions are used for all measurements

1. Download the required models (`yolo8x.pt`, `best.pt`, `last.pt`)
2. Place the downloaded models in the `models/` directory

3. **Validation**:
   - Checks if points are inside the field
   - Validates transformed coordinates
   - Ensures accurate measurements

## Notes

- The system supports caching of intermediate results in the `stubs/` directory
- Processing time depends on the video length and resolution
- GPU acceleration is recommended for optimal performance
- Speed calculations are smoothed using a 5-frame window
- Unrealistic speeds (>40 km/h) are capped and logged
- Field calibration is hardcoded for consistent measurements
