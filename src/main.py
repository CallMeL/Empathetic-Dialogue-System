import argparse
import os
import logging
from model import StrokeModel
from utils import load_data, preprocess_data


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train and evaluate a machine learning model.")
     
    return parser.parse_args()


def setup_logging(log_file):
    logging.basicConfig(filename=log_file, level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Logging setup complete.")


def main():

    args = parse_arguments()

    model = StrokeModel()

    logging.info("Training model...")
    #model.fit(X_train, y_train)
    logging.info("Complete training model")

if __name__ == "__main__":
    main()
    print("hello world")
