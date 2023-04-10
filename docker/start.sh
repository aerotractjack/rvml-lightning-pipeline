#!/bin/bash

# Default values for new image name and tag
IMAGE="raster-vision-run-python"
TAG="latest"

# Parse arguments
while [[ $# -gt 0 ]]; do
	case $1 in
		-i|--image)
			IMAGE=$2
			shift
			shift
			;;
		-t|--tag)
			TAG=$2
			shift
			shift
			;;
	esac
done

# Remove old duplicate images and build a new one
docker rmi -f $IMAGE:$TAG
docker build --no-cache --tag=$IMAGE:$TAG .
sleep 1

# Create the `docker run` command
read -r -d '' CMD <<- EOF 
docker run
	--gpus=all
	-e AWS_PROFILE=${AWS_PROFILE:-default} -v ${HOME}/.aws:/root/.aws:ro
	-p 7117:7177
	${IMAGE}
EOF

# Display to user and execute
echo $CMD
$CMD
