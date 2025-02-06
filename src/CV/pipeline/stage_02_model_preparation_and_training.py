from CV.config.configuration import ConfigurationManager
from CV.components.model_preparation_and_training import YOLOModelTrainer
from CV import logger



STAGE_NAME = "Model preparation and training"


class ModelPreparationAndTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_preparation_and_training_config = config.get_model_preparation_and_training_config()
        model_preparation_and_training = YOLOModelTrainer(config=model_preparation_and_training_config)
        model_preparation_and_training.prepare_and_train_yolo_model()



if __name__ == '__main__':
    import torch.multiprocessing as mp
    mp.freeze_support()  
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelPreparationAndTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
