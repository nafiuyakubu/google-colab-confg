Installation
You can install the TensorFlow Object Detection API either with Python Package Installer (pip) or Docker. For local runs we recommend using Docker and for Google Cloud runs we recommend using pip.

Clone the TensorFlow Models repository and proceed to one of the installation options.

git clone https://github.com/tensorflow/models.git
Docker Installation

# From the root of the git repository

docker build -f research/object_detection/dockerfiles/tf2/Dockerfile -t od .
docker run -it od

##################################################################

## Python Package Installation

cd models/research

# Compile protos.

protoc object_detection/protos/\*.proto --python_out=.

# Install TensorFlow Object Detection API.

cp object_detection/packages/tf2/setup.py .
python -m pip install --use-feature=2020-resolver .

# Test the installation.

python object_detection/builders/model_builder_tf2_test.py
##################################################################

echo "Cython<3" > cython_constraint.txt
PIP_CONSTRAINT=cython_constraint.txt python -m pip install --use-feature=fast-deps .

python3 -m pip install --use-feature=fast-deps .
