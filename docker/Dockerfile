FROM nvcr.io/nvidia/pytorch:23.04-py3

RUN apt-get update && apt-get install -y apt-utils && rm -rf /var/lib/apt/lists/*

# Set the locale
RUN  apt-get update && apt-get install locales && locale-gen en_US.UTF-8 && rm -rf /var/lib/apt/lists/*
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

# Set the OS
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    tzdata wget mlocate build-essential openssh-client nano cron yasm pkg-config cmake libjpeg-dev libpng-dev ffmpeg \
    libtiff-dev libavcodec-dev libavformat-dev autossh libswscale-dev libv4l-dev libxvidcore-dev libx264-dev libsm6 \
    libgtk-3-dev libatlas-base-dev gfortran libpq-dev libxext6 && rm -rf /var/lib/apt/lists/*

# set python
RUN pip3 install psutil Pillow numpy python-crontab opencv-python jupyterlab django websockets uvicorn gunicorn \
    django-cors-headers ultralytics djangorestframework markdown django-filter psycopg2

# set uvicorn
RUN sed -i  '/config.should_reload:/a  \ \ \ \ \ \ \ \ server.force_exit = True' /usr/local/lib/python3.8/dist-packages/uvicorn/main.py

# set the app
WORKDIR /App
RUN date +%s >/App/build-date.txt
COPY docker/start.sh /App
ADD https://api.github.com/repos/aratech-IA/Api_IA/commits/main latest_commit
RUN git clone https://github.com/Aratech-IA/Api_IA.git

CMD ["/App/start.sh"]
