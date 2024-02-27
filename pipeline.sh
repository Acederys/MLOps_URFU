pip install kaggle
cp kaggle.json ~/.kaggle/kaggle.json
chmod 600 kaggle.json
kaggle datasets download -d abrahamanderson/loan-data
unzip loan-data.zip
python3 data_creation.py
python3 model_preprocessing.py
python3 model_preparation.py
python3 model_testing.py