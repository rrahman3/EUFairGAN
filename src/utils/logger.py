import os
import csv

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

def write_to_csv(file_name, data):
    # Open the file in append mode so that data is added to the end of the file
    with open(file_name, mode='a', newline='\n') as file:
        writer = csv.writer(file)
        # Write the data row
        writer.writerow(data)

import random
# Set the seed
random.seed(2024)


import datetime
from pathlib import Path
now = datetime.datetime.now()
timestamp_str = now.strftime("%Y%m%d_%H%M%S")

# utils.get_directory_to_save_file(folder_name='FineTunedESRGAN', file_name=f'image_{epoch:05d}.jpg', type='images')
def get_directory_to_save_file(folder_name, file_name, type="models"):#models/images/results/
    if type == 'results':
        directory = f'outputs/{folder_name}'
        file_path = f'{directory}/{timestamp_str}_{file_name}'

    else:
        directory = f'outputs/{folder_name}_{timestamp_str}/{type}'
        file_path = f'{directory}/{file_name}'
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return file_path

