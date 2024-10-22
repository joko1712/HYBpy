FROM certbot/certbot:v1.3.0

ENV APP_HOME=/home/hybpy

RUN apk update && apk add bash curl

RUN mkdir -p /var/www && mkdir /var/www/certbot
RUN mkdir -p /etc/letsencrypt
RUN mkdir -p $APP_HOME

COPY ./setup.sh $APP_HOME
RUN chmod +x $APP_HOME/setup.sh
