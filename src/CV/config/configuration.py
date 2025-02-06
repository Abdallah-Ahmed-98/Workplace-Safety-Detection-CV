import os
from CV.constants import *
from CV.utils.common import read_yaml, create_directories
from CV.entity.config_entity import (DataIngestionConfig,
                                                ModelPreparationAndTrainingConfig,
                                                EvaluationConfig)


class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])


    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir, 
            dataset_name=config.dataset_name,
            data_yaml_file=config.data_yaml_file
        )
        
        return data_ingestion_config
    

    def get_model_preparation_and_training_config(self) -> ModelPreparationAndTrainingConfig:
        config = self.config.model_preparation_and_training
        params = self.params

        create_directories([config.root_dir])

        model_preparation_and_training_config = ModelPreparationAndTrainingConfig(
            root_dir=Path(config.root_dir),
            data_yaml_file=Path(config.data_yaml_file),
            yolo_model=Path(config.yolo_model),
            params_device=params.DEVICE,
            params_image_size=params.IMAGE_SIZE,
            params_epochs=params.EPOCHS,
            params_batch_size=params.BATCH_SIZE,
            params_learning_rate_0=params.LEARNING_RATE_0,
            params_learning_rate_f=params.LEARNING_RATE_F,
            params_augmentation=params.AUGMENTATION,
            params_augmentation_hyperparams=params.AUGMENTATION_HYPERPARAMS,
        )

        return model_preparation_and_training_config


    def get_evaluation_config(self) -> EvaluationConfig:

        evaluation = self.config.evaluation
        
        create_directories([
            Path(evaluation.root_dir)
        ])

        eval_config = EvaluationConfig(
            root_dir=Path(evaluation.root_dir),
            data_yaml_file=Path(evaluation.data_yaml_file),
            best_model=Path(evaluation.best_model),
        )
        return eval_config