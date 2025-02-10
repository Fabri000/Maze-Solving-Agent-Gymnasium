import os
import logging
from datetime import datetime

def init_logger(log_name:str,log_dir:str):
    log_dir = log_dir
    os.makedirs(log_dir, exist_ok=True)

    file_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    logging.basicConfig(filename=f"{log_dir}/run_{file_name}.log",filemode="a",format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',level= logging.DEBUG)
    logger = logging.getLogger(log_name)
    return logger