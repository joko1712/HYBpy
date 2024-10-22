FROM nginx:1.17.4-alpine
ENV APP_HOME=/home/hybpy

RUN rm /etc/nginx/conf.d/default.conf
COPY nginx.conf /etc/nginx/conf.d
COPY hybpy_frontend.conf /etc/nginx
COPY hybpy_api.conf /etc/nginx
COPY ssl.conf /etc/nginx

# Create the appropriate directories
RUN mkdir $APP_HOME
WORKDIR $APP_HOME
RUN mkdir $APP_HOME/static
RUN mkdir /var/www && mkdir /var/www/certbot
RUN mkdir /etc/letsencrypt
