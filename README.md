# Api IA

This is the project to make api with IA service.

## Objects recognition

API server to use Ultralytics yolov8

https://github.com/ultralytics/ultralytics

### Specific notice about making the automatic migrations when running in remote dev.

Because noting is running in local, you don't have any container with the database on your local pc. So you can not run the migrations in local.
You have to connect to the your dev remote instance to make the migrations. To do so, you have to run the following command:

    ssh -p 2223 user@ia.aratech.cloud
    docker ps # to find the container id
    docker exec -it <container id> bash

Then you can run the migrations:

    python manage.py makemigrations


