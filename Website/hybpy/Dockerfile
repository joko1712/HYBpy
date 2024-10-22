FROM node:18 as node_frontend
ENV APP_HOME=/home/hybpy

ENV DASH_DEBUG_MODE False

WORKDIR $APP_HOME/frontend

# My Code
COPY . ./

RUN set -ex && \
    npm install && npm run build

ENV DEV False

EXPOSE 3000

# CMD ["sh", "-c", "npm start -- --port 8050"]

CMD ["npm", "start"]