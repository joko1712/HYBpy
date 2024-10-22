# THIS DOCKERFILE SHOULD BE INSIDE THE BACKEND FOLDER!
FROM python:3.9
ENV APP_HOME=/home/hybpy

ENV DEV False

WORKDIR $APP_HOME

# My Code
COPY . ./

WORKDIR $APP_HOME/Backend

RUN set -ex && \
    pip install -r requirements.txt && pip install gunicorn

EXPOSE 5000

CMD ["gunicorn","--bind", "0.0.0.0:5000", "-w", "17", "--timeout", "7200", "app:app"]