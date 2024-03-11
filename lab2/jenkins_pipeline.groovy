pipeline {
    agent any

    stages {
        stage('Git') {
            steps {
                sh 'sudo rm -rf *'
                git branch: 'module2', url: 'https://github.com/Acederys/MLOps_URFU.git'
            }
        }
        stage('Project preparation') {
            steps {
                sh '''cd ./lab2
                ls
                python3 -m venv ./venv
                . ./venv/bin/activate
                pip install -r ./requirements.txt
                mkdir ./test ./train'''
            }
        }
        stage('Data creation') {
            steps {
                sh '''cd ./lab2
                rm -rf ~/.kaggle
                mkdir ~/.kaggle
                cp kaggle.json ~/.kaggle/kaggle.json
                chmod 600 ~/.kaggle/kaggle.json
                kaggle datasets download -d waalbannyantudre/south-african-heart-disease-dataset
                unzip south-african-heart-disease-dataset.zip
                python3 data_creation.py'''
            }
        }
        stage('Model preprocessing') {
            steps {
                sh '''cd ./lab2
                python3 model_preprocessing.py'''
            }
        }
        stage('Model preparation') {
            steps {
                sh '''cd ./lab2
                python3 model_preparation.py'''
            }
        }
        stage('Model testing') {
            steps {
                sh '''cd ./lab2
                python3 model_testing.py'''
            }
        }
    }
}