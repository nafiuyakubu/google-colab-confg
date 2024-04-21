### Bash script to Setup my Enviroment.
# Author: Nafiu Yakubu [Software Engineer, Full Stack Developer, Data Analyst, Database Engineer, AI/ML, DevOps, IaC,]
# Date: 4/21/2024

#!/bin/bash

############------[Step 1: Variable Setup]-------############
working_dir=$(pwd)
base_dir="dataset"
images_dir="${base_dir}/images"
models_dir="${base_dir}/models"
file_url="https://www.dropbox.com/s/gk57ec3v8dfuwcp/CoinPics_11NOV22.zip?dl=1"
label_map_file="${base_dir}/labelmap.txt"
train_record_file="${base_dir}/train.tfrecord"
val_record_file="${base_dir}/val.tfrecord"
label_map_pbtxt_file="${base_dir}/labelmap.pbtxt"
model_config_file="models_config.json"
chosen_model='ssd-mobilenet-v2-fpnlite-320'
num_steps=40000
batch_size=16
# Replace these variables with your repository URL, file path, and access token
REPO_URL="https://github.com/nafiuyakubu/google-colab-confg" 
FILE_PATH="path/to/your/python_file.py"
ACCESS_TOKEN="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"
download_file_from_repo() {
    local FILE_NAME="${3##*/}" && echo $FILE_NAME  # Extract filename from path
    if [ -e "$FILE_PATH" ]; then
        echo "File '$FILE_NAME' already exists Locally. Skipping download."
    else
        #$1[ACCESS_TOKEN] $2[REPO_URL] $3[FILE_PATH]
        curl -H "Authorization: token $1"  -H "Accept: application/vnd.github.v3.raw"  -o "$3"  "$2/raw/main/$3"
        echo "File '$FILE_NAME' downloaded successfully."
    fi
}
# download_file_from_repo "$ACCESS_TOKEN" "$REPO_URL" "SampleTFSSDModelTraining/img_dataset_spliter.py"
# label_csv="${base_dir}/labelmap.txt"

############------[Step 1: Install TensorFlow Object Detection Dependencies]-------############
pip uninstall Cython -y # Temporary fix for "No module named 'object_detection'" error
# -----------------(Download)----------------- #
repo_url="https://github.com/tensorflow/models" # Define the repository URL
clone_dir="${models_dir}" # Define Destination directory 
# Check if the directory already exists
if [ ! -d "$clone_dir" ]; then
    # If the directory does not exist, clone the repository
    set -e  # Enable Exit immediately if any command exits with a non-zero
    git clone --depth 1 "$repo_url" "$clone_dir"
    set +e  # Disable the immediate exit to allow further commands to run
else
    echo "Directory already exists. Skipping cloning."
fi

# -----------------(Installation)----------------- #
cd "${base_dir}/models/research/"
protoc object_detection/protos/*.proto --python_out=. # Compile protos.
cp object_detection/packages/tf2/setup.py . # Copy setup files into models/research folder
# python -m pip install --use-feature=2020-resolver . # Install TensorFlow Object Detection API.
cd "$working_dir" # Return back to the working directory
# python "$base_dir/models/research/object_detection/builders/model_builder_tf2_test.py" # Test the installation.(Run Model Bulider Test file)


# For Colab[Install the Object Detection API (NOTE: This block takes about 10 minutes to finish executing)]
# Need to do a temporary fix with PyYAML because Colab isn't able to install PyYAML v5.4.1
pip install pyyaml==5.3
pip install "${base_dir}/models/research/"
# Need to downgrade to TF v2.8.0 due to Colab compatibility bug with TF v2.10 (as of 10/03/22)
pip install tensorflow==2.8.0
# Install CUDA version 11.0 (to maintain compatibility with TF v2.8.0)
pip install tensorflow_io==0.23.1
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget http://developer.download.nvidia.com/compute/cuda/11.0.2/local_installers/cuda-repo-ubuntu1804-11-0-local_11.0.2-450.51.05-1_amd64.deb
dpkg -i cuda-repo-ubuntu1804-11-0-local_11.0.2-450.51.05-1_amd64.deb
apt-key add /var/cuda-repo-ubuntu1804-11-0-local/7fa2af80.pub
apt-get update && sudo apt-get install cuda-toolkit-11-0
export LD_LIBRARY_PATH=/usr/local/cuda-11.0/lib64:$LD_LIBRARY_PATH
python "$base_dir/models/research/object_detection/builders/model_builder_tf2_test.py" # Test the installation.(Run Model Bulider Test file)



############------[Step 1: Create Directories]-------############
# Create directories if they do not exist
# mkdir -p dataset/models && mkdir -p dataset/images/train && mkdir -p dataset/images/validation && mkdir -p dataset/images/test
mkdir -p "$base_dir/"
mkdir -p "$models_dir/"
mkdir -p "$models_dir/mymodel"
mkdir -p "$images_dir/"
mkdir -p "$images_dir/train"
mkdir -p "$images_dir/validation"
mkdir -p "$images_dir/test"


############------[Step 2: Prepare the Zip DataSet(Online/Offline)]-------############
# Check if the file exists before downloading
if [ ! -f "$base_dir/images.zip" ]; then
    echo "------Downloading.........------"
    # Download the zip file to the specified directory
    # wget -O "$base_dir/images.zip" "$file_url" 
    curl -L -o "$base_dir/images.zip" "$file_url"
    echo "------Downloading Completed------"
fi

# pip3 install /datasets/models/research/ 
# python3 /dataset/models/research/object_detection/builders/model_builder_tf2_test.py

Check if the file exists and unzip it if it does
if [ -f "$base_dir/images.zip" ]; then
    echo "------UnZipping Started .........------"
    unzip -o "$base_dir/images.zip" -d "$base_dir/images/all"
    echo "------UnZipping Completed------"
fi


############------[Step 3: Run my Custom Dataset Spliter(train/validation/test)]-------############
# Run Python script 
download_file_from_repo "$ACCESS_TOKEN" "$REPO_URL" "SampleTFSSDModelTraining/img_dataset_spliter.py"
python3 img_dataset_spliter.py \
    --image_path="$base_dir/images/all"  \
    --train_path="$base_dir/images/train"  \
    --val_path="$base_dir/images/validation"  \
    --test_path="$base_dir/images/test"


############------[Step 4: Create the Labelmap]-------############
: <<'END_COMMENT'
The code section below will create a "labelmap.txt" file that contains a list of classes. 
Replace the class1, class2, class3 text with your own classes (for example, penny, nickel, dime, quarter), 
adding a new line for each class. Then, click play to execute the code
#Note [The label map is the separate source of record for class annotations. Hands on with the Label Map.]
END_COMMENT
# Append the contents to the labelmap.txt file
# Define the labels to append
label_list=("penny", "nickel", "dime", "quarter")
# Loop through each new label
for label in "${label_list[@]}"; do
    # Check if the label already exists in the labelmap.txt file
    if ! grep -qF "$label" "${base_dir}/labelmap.txt"; then
        # Append the label to the labelmap.txt file
        echo "$label" >> "${base_dir}/labelmap.txt"
    fi
done
########[Optionally Just Add]#######
# cat <<EOF >> "${base_dir}/labelmap.txt"
# class1
# class2
# class3
# EOF


############------[Step 5: Create CSV data files and TFRecord files]-------############
: <<'END_COMMENT'
The TFRecord format is a simple format for storing a sequence of binary records.
END_COMMENT
download_file_from_repo "$ACCESS_TOKEN" "$REPO_URL" "SampleTFSSDModelTraining/create_csv.py"
download_file_from_repo "$ACCESS_TOKEN" "$REPO_URL" "SampleTFSSDModelTraining/create_tfrecord.py"

python3 create_csv.py
python3 create_tfrecord.py \
    --csv_input="$base_dir/images/train_labels.csv" \
    --labelmap="$label_map_file" \
    --image_dir="$base_dir/images/train" \
    --work_dir="$base_dir" \
    --output_path="$train_record_file"

python3 create_tfrecord.py \
    --csv_input="$base_dir/images/validation_labels.csv" \
    --labelmap="$label_map_file" \
    --image_dir="$base_dir/images/validation" \
    --work_dir="$base_dir" \
    --output_path="$val_record_file"


############------[Step 6: Setup(Training Configuration) Download the pre-trained model weights and configuration]-------############
download_file_from_repo "$ACCESS_TOKEN" "$REPO_URL" "SampleTFSSDModelTraining/$model_config_file"
model_name=$(jq -r ".[\"$chosen_model\"].model_name" "$model_config_file")
pretrained_checkpoint=$(jq -r ".[\"$chosen_model\"].pretrained_checkpoint" "$model_config_file")
base_pipeline_file=$(jq -r ".[\"$chosen_model\"].base_pipeline_file" "$model_config_file")
# echo "MODEL INFO - name[$model_name]"  && echo " - pretrained_checkpoint[$pretrained_checkpoint]" && echo " - base_pipeline_file[$base_pipeline_file]"

# Download pre-trained model weights
download_tar="http://download.tensorflow.org/models/object_detection/tf2/20200711/$pretrained_checkpoint"
echo "Downloading pre-trained model weights: $download_tar"
curl -L -o "$models_dir/mymodel/$pretrained_checkpoint" "$download_tar"

# Extract pre-trained model weights
echo "Extracting pre-trained model weights: $pretrained_checkpoint"
tar -zxvf "$models_dir/mymodel/$pretrained_checkpoint" -C "$models_dir/mymodel/"
# tar -zxvf "$models_dir/$pretrained_checkpoint"

# Download training configuration file for model
download_config="https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/configs/tf2/$base_pipeline_file"
echo "Downloading training configuration file: $download_config"
curl -L -o "$models_dir/mymodel/$base_pipeline_file" "$download_config"

download_file_from_repo "$ACCESS_TOKEN" "$REPO_URL" "SampleTFSSDModelTraining/start_training.py"
python3 start_training.py \
    --chosen_model="$model_name" \
    --model_name="$model_name" \
    --num_steps="$num_steps" \
    --batch_size="$batch_size" \
    --pretrained_checkpoint="$pretrained_checkpoint" \
    --base_pipeline_file="$base_pipeline_file" \
    --label_map_pbtxt_file="$label_map_pbtxt_file" \
    --train_record_file="$train_record_file" \
    --val_record_file="$val_record_file"