# Practices for MLops course UrFU

## Module 1
Сведения

Необходимо из создать простейший конвейер для автоматизации работы с моделью машинного обучения.

Отдельные этапы конвейера машинного обучения описываются в разных python–скриптах, которые потом соединяются в единую цепочку действий с помощью bash-скрипта.

Все файлы необходимо разместить в подкаталоге lab1 корневого каталога

Этапы:

1. Создайте python-скрипт (data_creation.py), который создает различные наборы данных, описывающие некий процесс. Таких наборов должно быть несколько, в некоторые данные можно включить аномалии или шумы. Часть наборов данных должны быть сохранены в папке “train”, другая часть в папке “test”. Одним из вариантов выполнения этого этапа может быть скачивание набора данных из сети, и разделение выборки на тестовую и обучающую. Учтите, что файл должен быть доступен и методы скачивания либо есть в ubuntu либо устанавливаются через pip в файле pipeline.sh
2. Создайте python-скрипт (data_preprocessing.py), который выполняет предобработку данных. Трансформации выполняются и над тестовой и над обучающей выборкой.
3. Создайте python-скрипт (model_preparation.py), который создает и обучает модель машинного обучения на построенных данных из папки “train”.
4. Создайте python-скрипт (model_testing.py), проверяющий модель машинного обучения на построенных данных из папки “test”.
5. Напишите bash-скрипт (pipeline.sh), последовательно запускающий все python-скрипты. При необходимости усложните скрипт. В результате выполнения скрипта на терминал в стандартный поток вывода печатается одна строка с оценкой метрики на вашей модели
   
### Запуск

Загрузите в каталог `kaggle.json`. Инструкцию можно посмотреть [здесь](https://www.kaggle.com/docs/api).

Запустите ./pipeline.sh

### Описание датасета

Датасет взят из [kaggle](https://www.kaggle.com/datasets/itssuru/loan-data), использованы данные о кредитовании за 2007-2010 годы для модели классификации.

Описание столбцов и замены пропусков

| Столбец  | Данные | Тип | Наличие пропусков  | Способ замены | Способ преобразования |
| ------------- | ------------- | ------------ | ------------- |------------ | ------------- |
| Gender  | Male/Female | категориальный  | 5 | Male  |  OrdinalEncoder |
|  Married  | Yes/No  |  категориальный  | 0  | -  |  OrdinalEncoder |
| Dependents  | 0/1/2/3+  | категориальный  | 8  | 0  | OrdinalEncoder  |
| Education  | Graduate/Not Graduate  | категориальный  | 0  | -  | OrdinalEncoder  |
| Self_Employed  | Yes/No | категориальный  | 21  | No  | OrdinalEncoder  |
| ApplicantIncome | 150-9703  | числовые  | 0  | -  | StandardScaler  |
| CoapplicantIncome  | 0-33837  | числовые  | 0  | -  | StandardScaler  |
| LoanAmount  | 9-150  | числовые  | 0  | -  | StandardScaler  |
| Loan_Amount_Term  | 12-480  | числовые  | 11  | 360  | StandardScaler  |
| Credit_History  | 1/0  | категориальный  | 30  | 0  | OrdinalEncoder  |
| Property_Area  | Rural/Urban/Semiurban	 | категориальный  | 0  | -  | OrdinalEncoder  |
| Loan_Status (целевой)  | Y/N  | категориальный  | 0 | -  | LabelEncoder  |

Для обучения модели использована Логистическая регрессия. Вывод осуществляется в консоль.

`Accuracy: 0.753 || Precision: 0.794 || Recall: 0.893 || F1 score: 0.84`


## Module 2
Сведения

Конвеер с использованием Jenkins
Срок заканчивается 18 марта 2024 г., 23:59
Инструкции
Нужно разработать собственный конвейер автоматизации для проекта машинного обучения. Для этого понадобится виртуальная машина с установленным Jenkins, python и необходимыми библиотеками. В ходе выполнения практического задания вам необходимо автоматизировать сбор данных, подготовку датасета, обучение модели и работу модели. 
 
Этапы задания 
1.	Развернуть сервер с Jenkins, установить необходимое программное обеспечение для работы над созданием модели машинного обучения. 
2.	Выбрать способ получения данных (скачать из github, из Интернета, wget, SQL запрос, …). 
3.	Провести обработку данных, выделить важные признаки, сформировать датасеты для тренировки и тестирования модели, сохранить. 
4.	Создать и обучить на тренировочном датасете модель машинного обучения, сохранить в pickle или аналогичном формате. 
5.	Загрузить сохраненную модель на предыдущем этапе и проанализировать ее качество на тестовых данных. 

   
### Запуск

Загрузите в каталог `kaggle.json`. Инструкцию можно посмотреть [здесь](https://www.kaggle.com/docs/api).

Запустите ./pipeline.sh

### Описание датасета

Датасет взят из [kaggle]([https://www.kaggle.com/datasets/itssuru/loan-data](https://www.kaggle.com/datasets/waalbannyantudre/south-african-heart-disease-dataset)), использованы данные о сердечно-сосудистых заболеваниях.

Описание столбцов 


sbp	      - Систолическое артериальное давление
tobacco	  -  Совокупный табак (кг)
ldl	      -  Уровень холестерина липопротеинов низкой плотности
adiposity	-  Тяжелый избыточный вес (числовой вектор)
famhist	  -  Семейный анамнез сердечно-сосудистых заболеваний
typea	    -  Поведение типа А
obesity	  -  Чрезмерное накопление жира (числовой вектор)
alcohol	  -  Текущее потребление алкоголя
age	      -  Возраст начала
chd	      -  Ответ, ишемическая болезнь сердца


![MLops2](https://github.com/Acederys/MLOps_URFU/assets/139765792/613fb3b3-d82b-44dc-a06c-170c885f21c6)


Для обучения модели использована Логистическая регрессия.
Результат:


![image](https://github.com/Acederys/MLOps_URFU/assets/139765792/743deccc-c69c-4dba-a89e-383d918ff621)

## Module 3
Сведения

В практическом задание по модулю вам необходимо применить полученные знания по работе с docker (и docker-compose). Вам необходимо использовать полученные ранее знания по созданию микросервисов. В этом задании необходимо развернуть микросервис в контейнере докер.
 
Этапы задания 
1.	Подготовить python код для модели и микросервиса
2.	Создать Docker file
3.	Создать docker образ
4.	Запустить docker контейнер и проверить его работу 
   
### Запуск

#### Если не устанволен docker 

Add Docker's official GPG key:

`sudo apt-get update`

`sudo apt-get install ca-certificates curl`

`sudo install -m 0755 -d /etc/apt/keyrings`

`sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc`

`sudo chmod a+r /etc/apt/keyrings/docker.asc`

Add the repository to Apt sources:

`echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null`
  
`sudo apt-get update`

#### Установите пакеты Docker

`sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin`

#### Сборка образа

`docker build -t app:latest -f Dockerfile .`

Найдем созданный образ:

`docker images | grep app`

####  Запуск образа

`docker run -p 8501:8501 -d app`

#### Остановка контейнера

Смотрим номера запущенных контейнеров

`docker ps`

Останавливаем контейнер

`docker stop {номер конейнера}`

#### Использование docker-compose

Для установки *docker-compose* выполняем команду:

`sudo apt-get update
sudo apt-get install docker-compose`

В корневой директории проекта создаем файл *docker-compose.yml* со следующим содержанием:

`
services:
  streamlit-app:
    build: .
    ports:
      - "8501:8501"
`

Запускаем docker-compose при помощи команды:

`docker-compose up`

#### Dickerhub

Образ выгружен на dockerhub: 

https://hub.docker.com/repository/docker/acederus/mlops/general

#### Модель для классификации изображений

Модель взяла с hugging Face: `google/vit-base-patch16-224`


