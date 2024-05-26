# example deep learning VM
# for a full list see us-docker.pkg.dev/deeplearning-platform-release/gcr.io/
# and for details see https://cloud.google.com/deep-learning-vm/docs/images#supported-frameworks
FROM us-docker.pkg.dev/deeplearning-platform-release/gcr.io/pytorch-gpu.2-2.py310

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE 1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED 1

# pip gives a warning if you install packages as root
# set this flag to just ignore the warning
ENV PIP_ROOT_USER_ACTION=ignore

RUN pip install -U pip
WORKDIR /workspace

# install other requirements
COPY requirements.txt .
RUN pip install -r requirements.txt

# copy the rest of the files into the container
COPY src .

# Define environment variable
ENV MODEL_PATH=/workspace/models

# start model service
CMD uvicorn api_service:app --port 5004 --host 0.0.0.0

# docker build -t averagervctrainer-vlm .
# docker run -p 5004:5004 --gpus all -d averagervctrainer-vlm
# docker kill CONTAINER-ID
# docker tag averagervctrainer-vlm asia-southeast1-docker.pkg.dev/dsta-angelhack/repository-averagervctrainer/averagervctrainer-vlm:latest
# docker push asia-southeast1-docker.pkg.dev/dsta-angelhack/repository-averagervctrainer/averagervctrainer-vlm:latest
# gcloud ai models upload --region asia-southeast1 --display-name 'averagervctrainer-vlm' --container-image-uri asia-southeast1-docker.pkg.dev/dsta-angelhack/repository-averagervctrainer/averagervctrainer-vlm:latest --container-health-route /health --container-predict-route /identify --container-ports 5004 --version-aliases default