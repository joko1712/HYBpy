#!/bin/bash

# Original script: https://raw.githubusercontent.com/wmnnd/nginx-certbot/master/init-letsencrypt.sh

if [ -z "${DEV}" ]; then
    echo "Environmental variables are missing!"
    exit 1
fi

domains=(hybpy.com www.hybpy.com api.hybpy.com www.api.hybpy.com)
rsa_key_size=4096
conf_path="/etc/letsencrypt"
www_path="/var/www/certbot"
email="dmgonca@gmail.com"

mkdir -p "$conf_path/live"
domain_path="$conf_path/live/$domains"
mkdir -p "$domain_path"

if [ ! -f "$conf_path/options-ssl-nginx.conf" ] || [ ! -f "$conf_path/ssl-dhparams.pem" ]; then
    echo "### Downloading recommended TLS parameters ..."
    curl -s https://raw.githubusercontent.com/certbot/certbot/master/certbot-nginx/certbot_nginx/_internal/tls_configs/options-ssl-nginx.conf > "$conf_path/options-ssl-nginx.conf"
    curl -s https://raw.githubusercontent.com/certbot/certbot/master/certbot/certbot/ssl-dhparams.pem > "$conf_path/ssl-dhparams.pem"
fi

# Existing data found for $domains. Continuing replaces existing certificate.
if [ ! -z "$(ls -A $domain_path)" ]; then
    echo "### Existing data found for $domains. Not replacing certificates."
    trap exit TERM; while :; do certbot renew; sleep 12h & wait $!; done;
fi

echo "### Creating dummy certificate for $domains ..."
openssl req -x509 -nodes -newkey rsa:1024 -days 1\
    -keyout "$domain_path/privkey.pem" \
    -out "$domain_path/fullchain.pem" \
    -subj "/CN=localhost"

if [ $DEV == true ]; then
    trap exit TERM; while :; do certbot renew; sleep 12h & wait $!; done;
fi

echo "Wait for Nginx to start properly first time";
sleep 15s;

# Nginx will cache it's certificates
echo "### Deleting dummy certificate for $domains ..."
rm -Rf /etc/letsencrypt/live/$domains && \
    rm -Rf /etc/letsencrypt/archive/$domains && \
    rm -Rf /etc/letsencrypt/renewal/$domains.conf

echo "### Requesting Let's Encrypt certificate for $domains ..."
domain_args=""  # --> "-d hybpy.com -d www.hybpy.com -d api.hybpy.com -d www.api.hybpy.com"
for domain in "${domains[@]}"; do
  domain_args="$domain_args -d $domain"
done

# Select appropriate email arg
case "$SSL_EMAIL" in
  "") email_arg="--register-unsafely-without-email" ;;
  *) email_arg="--email $SSL_EMAIL" ;;
esac

certbot certonly \
    --webroot \
    --webroot-path $www_path \
    --non-interactive \
    $staging_arg \
    $email_arg \
    $domain_args \
    --rsa-key-size $rsa_key_size \
    --agree-tos \
    --force-renewal

sleep 30s;  # Wait for Certbot to be finished

echo "Force restarting Nginx from Certbot"
echo -e "POST /containers/$NGINX_CONTAINER/restart HTTP/1.0\r\n" | nc -U /tmp/docker.sock

trap exit TERM; while :; do certbot renew; sleep 12h & wait $!; done;
