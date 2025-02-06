from pathlib import Path
import os
from CV.entity.config_entity import EvaluationConfig
from ultralytics import YOLO


import numpy as np


class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    def evaluation(self):

        best_model_path = self.config.best_model

        # Load YOLO model
        try:
            best_model = YOLO(best_model_path)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Best model file not found at {best_model_path}") from e


        # Dynamically resolve the output directory
        base_dir = Path(os.getcwd()).resolve()  # Get the directory of the script
        output_dir = base_dir / self.config.root_dir

        # Ensure the output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)

        # Train the YOLO model
        try:
            # Train the model
            best_model.val(
                data=self.config.data_yaml_file,
                workers=0,  # Avoid multiprocessing issues
                project=str(output_dir),  # Convert Path to string for YOLOv8
                name="yolov8n_evaluation"  # Custom name for this run
            )
        except Exception as e:
                raise e