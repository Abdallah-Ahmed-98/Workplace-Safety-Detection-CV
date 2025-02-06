import os
import zipfile
import yaml
import gdown
from CV import logger
from CV.entity.config_entity import DataIngestionConfig
from pathlib import Path


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config


    
     
    def download_file(self)-> str:
        '''
        Fetch data from the url
        '''

        try: 
            dataset_url = self.config.source_URL
            zip_download_dir = self.config.local_data_file
            os.makedirs("artifacts/data_ingestion", exist_ok=True)
            logger.info(f"Downloading data from {dataset_url} into file {zip_download_dir}")

            file_id = dataset_url.split("/")[-2]
            prefix = 'https://drive.google.com/uc?/export=download&id='
            gdown.download(prefix+file_id,zip_download_dir)

            logger.info(f"Downloaded data from {dataset_url} into file {zip_download_dir}")

        except Exception as e:
            raise e
        
    
    def extract_zip_file(self):
        """
        zip_file_path: str
        Extracts the zip file into the data directory
        Function returns None
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)




    def update_data_yaml_file(self):
        """
        Modify paths inside a data.yaml file.

        Returns:
            None: The function modifies the file in-place.
        """

        # Path to the data.yaml file
        data_yaml_path = Path("artifacts/data_ingestion/Workplace-Safety-Dataset/data.yaml")
        new_paths={
            "train": "artifacts/data_ingestion/Workplace-Safety-Dataset/train/images",
            "val": "artifacts/data_ingestion/Workplace-Safety-Dataset/valid/images",
            "test": "artifacts/data_ingestion/Workplace-Safety-Dataset/test/images"
            }

        # Load the data.yaml file
        if not data_yaml_path.is_file():
            raise FileNotFoundError(f"data.yaml file not found at {data_yaml_path}")



        with open(data_yaml_path, "r") as f:
            data_yaml = yaml.safe_load(f)


        # Modify the paths
        for key, new_path in new_paths.items():
            try:
                data_yaml[key] = new_path
            except Exception as e:
                raise e
    
        # Save the updated data.yaml
        with open(data_yaml_path, "w") as f:
            yaml.dump(data_yaml, f)
 
        with open(data_yaml_path, "r") as f:
            data_yaml = yaml.safe_load(f)
    
        # Resolve the paths for train and val
        train_path = Path(data_yaml["train"]).resolve()
        val_path = Path(data_yaml["val"]).resolve()
        test_path = Path(data_yaml["test"]).resolve()

        # Check if the directories exist
        if not train_path.is_dir():
            raise FileNotFoundError(f"Training directory not found: {train_path}")

        if not val_path.is_dir():
            raise FileNotFoundError(f"Validation directory not found: {val_path}")

        # Update the paths in the data.yaml dictionary
        data_yaml["train"] = str(train_path)
        data_yaml["val"] = str(val_path)
        data_yaml["test"] = str(test_path)

        # Save the updated data.yaml
        with open(data_yaml_path, "w") as f:
            yaml.dump(data_yaml, f)