#!/bin/bash

# Define the correct path
correct_path="/home/hugomorchao/HYBpy"

# Get the current path
current_path=$(pwd)

# Check if current path is the correct one
if [ "$current_path" != "$correct_path" ]; then
  echo "You are not in the correct path. Changing to $correct_path..."
  cd "$correct_path" || { echo "Failed to change directory! Exiting..."; exit 1; }
else
  echo "You are already in the correct path."
fi

# Stop and kill all running containers related to the HybPy application
docker kill hybpy_nginx_prod hybpy_certbot_prod hybpy_frontend_prod hybpy_api_prod
# Remove all images related to the HybPy application
docker image rm -f hybpy_nginx hybpy_certbot hybpy_hybpy_api hybpy_hybpy_frontend
# Rebuild and start the containers defined in the docker-compose.yml file
echo "Y" | docker-compose up --build -d --remove-orphans --force-recreate
