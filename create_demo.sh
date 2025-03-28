#!/bin/bash

# Create demo directory if it doesn't exist
mkdir -p demo

# Convert video to GIF
# -t 10 takes first 10 seconds
# fps=10 reduces file size
# scale=800:-1 sets width to 800px and maintains aspect ratio
# Using palettegen for better quality
ffmpeg -i output_videos/output_video.mp4 -t 10 -vf "fps=10,scale=800:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" -loop 0 demo/output_demo.gif

echo "Demo GIF created at demo/output_demo.gif"


