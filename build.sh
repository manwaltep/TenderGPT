#!/bin/bash

# Name of the image
IMAGE_NAME="tendergpt"

# Build the Docker image
docker build -t $IMAGE_NAME .

echo "Image has been successfully built with the name: $IMAGE_NAME"
