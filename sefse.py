import os
import tempfile
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()
with tempfile.TemporaryDirectory() as tmpdirname:
    api.dataset_download_file('username/dataset-slug', file_name='file.csv', path=tmpdirname)d.
    file_path = os.path.join(tmpdirname, 'file.csv')
    df = pd.read_csv(file_path)
    print(df.head())
