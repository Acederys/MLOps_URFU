pip install -r requirements.txt
cp kaggle.json ~/.kaggle/kaggle.json
chmod 600 kaggle.json
kaggle datasets download -d bhavikjikadara/loan-status-prediction
unzip loan-status-prediction.zip
python3 data_creation.py
python3 model_preprocessing.py
python3 model_preparation.py
python3 model_testing.py