from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path
    dataset_name: str
    data_yaml_file: Path



@dataclass(frozen=True)
class ModelPreparationAndTrainingConfig:
    root_dir: Path                      
    data_yaml_file: Path
    yolo_model: Path
    params_device: str
    params_image_size: List[int]   
    params_epochs: int                  
    params_batch_size: int              
    params_learning_rate_0: float
    params_learning_rate_f: float  
    params_augmentation: bool             
    params_augmentation_hyperparams: dict               



@dataclass(frozen=True)
class EvaluationConfig:
    root_dir: Path                      
    data_yaml_file: Path
    best_model: Path