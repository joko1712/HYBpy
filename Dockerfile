# THIS DOCKERFILE SHOULD BE INSIDE THE BACKEND FOLDER!
FROM python:3.9-slim

ENV APP_HOME=/home/hybpy
ENV DEV=False

RUN apt-get update && apt-get install -y \
    build-essential \
    libhdf5-dev \
    python3-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR $APP_HOME

COPY . $APP_HOME/

WORKDIR $APP_HOME/Backend

RUN pip install --upgrade pip \
 && pip install -r requirements.txt \
 && pip install gunicorn

EXPOSE 5000

COPY hybpy-test-firebase-adminsdk-20qxj-fc73476cba.json /home/hybpy/Backend/

#CMD ["gunicorn","--bind", "0.0.0.0:5000", "-w", "17", "--timeout", "7200", "app:app"]
#CMD ["gunicorn", "--bind", "0.0.0.0:5000", "-w", "4", "--timeout", "7200", "app:app"]
#CMD ["gunicorn", "--bind", "0.0.0.0:5000", "-w", "17", "--timeout", "21600", "app:app"]
#CMD ["gunicorn", "--workers=2", "--bind", "0.0.0.0:8080", "app:app"]
#CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers=4", "--timeout=7200", "--certfile=/etc/letsencrypt/live/api.hybpy.com/fullchain.pem", "--keyfile=/etc/letsencrypt/live/api.hybpy.com/privkey.pem", "app:app"]
#CMD ["gunicorn", "-c", "gunicorn.conf.py", "app:app"]
#CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers=2", "--threads=4", "--timeout=7200", "--certfile=/etc/letsencrypt/live/api.hybpy.com/fullchain.pem", "--keyfile=/etc/letsencrypt/live/api.hybpy.com/privkey.pem", "--worker-class=gthread", "app:app"]
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers=4", "--timeout=7200", "--certfile=/etc/letsencrypt/live/api.hybpy.com/fullchain.pem", "--keyfile=/etc/letsencrypt/live/api.hybpy.com/privkey.pem", "app:app"]