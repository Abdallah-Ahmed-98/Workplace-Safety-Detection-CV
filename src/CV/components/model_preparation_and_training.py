import os
from ultralytics import YOLO
from CV.entity.config_entity import ModelPreparationAndTrainingConfig
from pathlib import Path

                                                

class YOLOModelTrainer:
    def __init__(self, config: ModelPreparationAndTrainingConfig):

        self.config = config

    def prepare_and_train_yolo_model(self):
        """
        Prepare and configure the YOLO model with the specified parameters.
        Train the YOLO model using the provided data configuration.
        """
        yolo_model_path = os.path.join(self.config.root_dir, self.config.yolo_model)

        # Load YOLO model
        try:
            yolo_model = YOLO(yolo_model_path)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"YOLO model file not found at {yolo_model_path}") from e

        # Apply learning rate
        yolo_model.overrides['lr0'] = self.config.params_learning_rate_0
        yolo_model.overrides['lrf'] = self.config.params_learning_rate_f

        # Conditionally apply augmentation hyperparameters
        if self.config.params_augmentation:
            yolo_model.overrides.update(self.config.params_augmentation_hyperparams)

        # Dynamically resolve the output directory
        base_dir = Path(os.getcwd()).resolve()  # Get the directory of the script
        output_dir = base_dir / self.config.root_dir

        # Ensure the output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)

        # Train the YOLO model
        try:
            # Train the model
            yolo_model.train(
                data=self.config.data_yaml_file,
                epochs=self.config.params_epochs,
                imgsz=self.config.params_image_size,
                batch=self.config.params_batch_size,
                device=self.config.params_device,
                workers=0,  # Avoid multiprocessing issues
                project=str(output_dir),  # Convert Path to string for YOLOv8
                name="yolov8n_training"  # Custom name for this run
            )

            # Delete the file at the specified path if it exists.
            file = Path("yolo11n.pt")

            if file.exists():
                file.unlink()  # Delete the file
                print(f"Deleted file: {file}")
            else:
                print(f"No file found at: {file}")
        
        except Exception as e:
                raise e