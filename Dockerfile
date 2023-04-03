FROM raster-vision-pytorch:latest
WORKDIR /app
COPY rvlightning.py ./
CMD [ "python3", "rvlightning.py"]
