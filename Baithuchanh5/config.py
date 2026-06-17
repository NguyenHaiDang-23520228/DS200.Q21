class Config:
    # Camera server -> Processing server
    processing_host = "localhost"
    processing_port = 6100

    # Processing server -> Storage server
    storage_host = "localhost"
    storage_port = 6200

    # Output directory for detection results
    results_dir = "output/results"

    # Camera settings
    camera_index = 0
    max_frames = 10
    frame_interval_sec = 0.5
    input_video = "input/sample.mp4"
    input_image_dir = "input/frames"
