# Football Computer Vision Analysis

A computer vision project that analyzes football/soccer videos to track players, the ball, and provide detailed game analytics.

## Demo

![Football Analysis Demo](demo/output_demo.gif)

Sample output showing:
- Player detection and tracking (colored boxes)
- Ball tracking (white box)
- Team classification (red vs blue boxes)
- Speed and distance measurements
- Camera movement visualization
- Ball possession tracking
- Player trajectories and heatmaps

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
- **Visual Annotations**: Overlay of tracking data and analytics on the video
- **Additional Visualizations**:
  - Player trajectories
  - Heatmaps
  - Field zones (defensive, midfield, attacking)

## Project Structure

```
football_cv_project/
├── input_videos/          # Input video files
├── output_videos/         # Processed output videos
├── models/               # YOLOv8 model files
├── stubs/                # Cached data for faster processing
├── calibration_results/  # Field transformation visualizations
├── camera_movement_estimator/  # Camera motion tracking
├── player_ball_assigner/      # Ball possession detection
├── speed_and_distance_estimator/  # Speed/distance calculations
├── team_assigner/        # Team classification
├── trackers/            # Object tracking implementation
├── utils/               # Utility functions
├── view_transformer/    # Perspective transformation
├── main.py             # Main execution script
└── README.md           # Project documentation
```

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

2. Run the analysis:
```bash
python main.py
```

3. The system will:
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

2. **Perspective Transform**:
   - Uses predefined field corners
   - Automatically validates transformed coordinates
   - Ensures measurements are within field bounds

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