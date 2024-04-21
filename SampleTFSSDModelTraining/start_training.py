
import re
import argparse
import subprocess


parser = argparse.ArgumentParser(description="Python Model Training Execution")
parser.add_argument("--chosen_model", help="Model Chosen", default="")
parser.add_argument("--model_name", help="Model Name", default="")
parser.add_argument("--num_steps", help="No of Steps", default=40000)
parser.add_argument("--batch_size", help="Batch Size", default=16)
parser.add_argument("--pretrained_checkpoint", help="", default="")
parser.add_argument("--base_pipeline_file", help="", default="")
parser.add_argument("--label_map_pbtxt_file", help="", default="")
parser.add_argument("--train_record_file", help="", default="")
parser.add_argument("--val_record_file", help="", default="")
args = parser.parse_args()



chosen_model = args.chosen_model
model_name = args.model_name
num_steps = args.num_steps
batch_size = args.batch_size
pretrained_checkpoint = args.pretrained_checkpoint
base_pipeline_file = args.base_pipeline_file
label_map_pbtxt_file = args.label_map_pbtxt_file
train_record_file = args.train_record_file
val_record_file = args.val_record_file





# Set file locations and get number of classes for config file
pipeline_fname = '/content/models/mymodel/' + base_pipeline_file
fine_tune_checkpoint = '/content/models/mymodel/' + model_name + '/checkpoint/ckpt-0'


'''
[NOTE:]
(num_steps): The total amount of steps to use for training the model. A good number to start with is 40,000 steps. 
You can use more steps if you notice the loss metrics are still decreasing by the time training finishes. 
The more steps, the longer training will take. Training can also be stopped early if loss flattens 
out before reaching the specified number of steps.
(batch_size): The number of images to use per training step. A larger batch size allows a model 
to be trained in fewer steps, but the size is limited by the GPU memory available for training. 
'''

# Set training parameters for the model
# num_steps = 40000
if chosen_model == 'efficientdet-d0':
  batch_size = 4
else:
  batch_size = 16




def get_num_classes(pbtxt_fname):
    from object_detection.utils import label_map_util
    label_map = label_map_util.load_labelmap(pbtxt_fname)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=90, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    return len(category_index.keys())

num_classes = get_num_classes(label_map_pbtxt_file)
print('Total classes:', num_classes)

# Create custom configuration file by writing the dataset, model checkpoint, and training parameters into the base pipeline file
def rewrite_pipeline_config_file():
    print('writing custom configuration file')
    with open(pipeline_fname) as f:
        s = f.read()

    with open('pipeline_file.config', 'w') as f:
        # Set fine_tune_checkpoint path
        s = re.sub('fine_tune_checkpoint: ".*?"',
                   'fine_tune_checkpoint: "{}"'.format(fine_tune_checkpoint), s)

        # Set tfrecord files for train and test datasets
        s = re.sub(
            '(input_path: ".*?)(PATH_TO_BE_CONFIGURED/train)(.*?")', 'input_path: "{}"'.format(train_record_file), s)
        s = re.sub(
            '(input_path: ".*?)(PATH_TO_BE_CONFIGURED/val)(.*?")', 'input_path: "{}"'.format(val_record_file), s)

        # Set label_map_path
        s = re.sub(
            'label_map_path: ".*?"', 'label_map_path: "{}"'.format(label_map_pbtxt_file), s)

        # Set batch_size
        s = re.sub('batch_size: [0-9]+',
                   'batch_size: {}'.format(batch_size), s)

        # Set training steps, num_steps
        s = re.sub('num_steps: [0-9]+',
                   'num_steps: {}'.format(num_steps), s)

        # Set number of classes num_classes
        s = re.sub('num_classes: [0-9]+',
                   'num_classes: {}'.format(num_classes), s)

        # Change fine-tune checkpoint type from "classification" to "detection"
        s = re.sub(
            'fine_tune_checkpoint_type: "classification"', 'fine_tune_checkpoint_type: "{}"'.format('detection'), s)

        # If using ssd-mobilenet-v2, reduce learning rate (because it's too high in the default config file)
        if chosen_model == 'ssd-mobilenet-v2':
            s = re.sub('learning_rate_base: .8',
                       'learning_rate_base: .08', s)

            s = re.sub('warmup_learning_rate: 0.13333',
                       'warmup_learning_rate: .026666', s)

        # If using efficientdet-d0, use fixed_shape_resizer instead of keep_aspect_ratio_resizer (because it isn't supported by TFLite)
        if chosen_model == 'efficientdet-d0':
            s = re.sub('keep_aspect_ratio_resizer', 'fixed_shape_resizer', s)
            s = re.sub('pad_to_max_dimension: true', '', s)
            s = re.sub('min_dimension', 'height', s)
            s = re.sub('max_dimension', 'width', s)

        f.write(s)


# %cd /content/models/mymodel

rewrite_pipeline_config_file()


# command = [
#     "python",
#     "/content/models/research/object_detection/model_main_tf2.py",
#     "--pipeline_config_path={}".format(pipeline_file),
#     "--model_dir={}".format(model_dir),
#     "--alsologtostderr",
#     "--num_train_steps={}".format(num_steps),
#     "--sample_1_of_n_eval_examples=1"
# ]

# # Execute the command
# subprocess.run(command, check=True)