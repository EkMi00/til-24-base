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

# start model service
CMD uvicorn api_service:app --port 5002 --host 0.0.0.0

# docker build -t averagervctrainer-nlp .
# docker run -p 5002:5002 --gpus all -d averagervctrainer-nlp
# docker kill CONTAINER-ID
# docker tag averagervctrainer-nlp asia-southeast1-docker.pkg.dev/dsta-angelhack/repository-averagervctrainer/averagervctrainer-nlp:latest
# docker push asia-southeast1-docker.pkg.dev/dsta-angelhack/repository-averagervctrainer/averagervctrainer-nlp:latest
# gcloud ai models upload --region asia-southeast1 --display-name 'averagervctrainer-nlp' --container-image-uri asia-southeast1-docker.pkg.dev/dsta-angelhack/repository-averagervctrainer/averagervctrainer-nlp:latest --container-health-route /health --container-predict-route /extract --container-ports 5002 --version-aliases default
