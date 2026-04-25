python 3.11.9

python manage.py makemigrations

# create database 
create database remote_sensing_db;
# migrate
python manage.py migrate

# train model 
python setup_training_data.py

python train_model.py --config_id 1 --dataset_id 1

it will take atleast 30-40 min 
# create superuser
python manage.py createsuperuser

# Run server
python manage.py runserver