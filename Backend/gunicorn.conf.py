import ssl

def when_ready(server):
    print("Gunicorn is ready. Server is listening...")

ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
ssl_ctx.minimum_version = ssl.TLSVersion.TLSv1_3
ssl_ctx.load_cert_chain(
    '/etc/letsencrypt/live/api.hybpy.com/fullchain.pem',
    '/etc/letsencrypt/live/api.hybpy.com/privkey.pem'
)

bind = "0.0.0.0:5000"
workers = 8
timeout = 7200
ssl_options = ssl_ctx
