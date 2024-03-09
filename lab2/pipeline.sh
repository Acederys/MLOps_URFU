mkdir test
mkdir  train
cp kaggle.json ~/.kaggle/kaggle.json
chmod 600 kaggle.json
kaggle datasets download -d waalbannyantudre/south-african-heart-disease-dataset
unzip south-african-heart-disease-dataset.zip
python3 data_creation.py
python3 model_preprocessing.py
python3 model_preparation.py
python3 model_testing.py