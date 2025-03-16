# THIS DOCKERFILE SHOULD BE INSIDE THE BACKEND FOLDER!
FROM python:3.9
ENV APP_HOME=/home/hybpy

ENV DEV False

WORKDIR $APP_HOME

# My Code
COPY . $APP_HOME/

WORKDIR $APP_HOME/Backend

RUN set -ex && \
    pip install -r requirements.txt && pip install gunicorn

EXPOSE 5000

#CMD ["gunicorn","--bind", "0.0.0.0:5000", "-w", "17", "--timeout", "7200", "app:app"]
#CMD ["gunicorn", "--bind", "0.0.0.0:5000", "-w", "4", "--timeout", "7200", "app:app"]
#CMD ["gunicorn", "--bind", "0.0.0.0:5000", "-w", "17", "--timeout", "21600", "app:app"]
#CMD ["gunicorn", "--workers=2", "--bind", "0.0.0.0:8080", "app:app"]
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers=4", "--timeout=7200", "--certfile=/etc/letsencrypt/live/api.hybpy.com/fullchain.pem", "--keyfile=/etc/letsencrypt/live/api.hybpy.com/privkey.pem", "--ssl-version TLSv1_3", "app:app"]
