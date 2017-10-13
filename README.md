# Object_Detection_Tensorflow

The most simplest steps for object detection in Python Tensorflow

## Installation
1. Install Tensorflow in a new Anaconda environment, follow the guide in Tensorflow offical page

2. Install the packages for Tensorflow

3. Download ```protoc_3.4.0```, put it in ```C:/```, add the environment path ```C:\protoc_3.4.0\bin``` to ```PATH```

    - Run the following code in command prompt
    
      ```python
      # From tensorflow/models/research/
      protoc object_detection/protos/*.proto --python_out=.
      ```
      
4. ADD a new environment path ```'PYTHONPATH'``` for both 'administrator' and 'system', add following variables
    
    ```
    C:\ProgramData\Anaconda3\envs\tensorflow\models\research\
    C:\ProgramData\Anaconda3\envs\tensorflow\models\research\slim\
    ```
    
## Execution
1. Run the code in object_detection_tutorial.ipynb

    https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

2. Install packages if missing

3. if run successfully, two images should be shown, with probabily and label on it

# Build your own category to pre-trained model
    
# (Transfer Learning)

After you successfully install and run the first model, you may build a new category to the pre-trained model, and train it again. This process called 'Transfer Learning'.

We need to prepare new images, label it, convert to specific format for tensorflow training.

## Data structure

Follow the structure below to set up the 'mac_and_cheese' category.

```
Object-Detection
-data/
--test_labels.csv
--train_labels.csv
-images/
--test/
---testingimages.jpg
--train/
---testingimages.jpg
--...yourimages.jpg
-training
-xml_to_csv.py
```

## Prepare image to tensorflow format
1. Download raw images with specific category (e.g. mac and cheese) from internet

2. Use ```labelImg``` to label the image, save the image to XML format

3. Seperate the data to **train** and **test** folder (90% : 10%)

4. Run ```xml_to_csv.py``` to convert XML to csv format
    - generate 'train_labels.csv' & 'test_labels.csv'

5. Run ```generate_tfrecord.py``` in command prompt to convert csv to tfrecord format
    ```
    python generate_tfrecord.py --csv_input=data/train_labels.csv --output_path=data/train.record
    
    python generate_tfrecord.py --csv_input=data/test_labels.csv --output_path=data/test.record
    ```
    - generate 'train.record' & 'test.record' 

## Config

1. Grab the config file from ```C:\ProgramData\Anaconda3\envs\tensorflow\models\research\object_detection\samples\configs```

2. Change the config properly (regarding to your data). You may use the config in this repo.

3. In ```training``` dir, add ```object-detection.pbtxt```

## Train the model

1. In ```mac_and_cheese``` folder, copy the file to ```\models\research\object_detection\```

2. In command prompt, change the directory to ```\models\research\object_detection\```

3. Run the following code:

    ```
    python train.py --logtostderr --train_dir=training --pipeline_config_path=training/ssd_mobilenet_v1_pets.config
    ```
4. Start the TensorBoard to see the loss

    ```
    tensorboard --logdir='training'
    ```
    
## Test the model

1. Generate the inference graph by ```export_inference_graph.py```, run this py with following code in commend prompt

    ```
    python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/ssd_mobilenet_v1_pets.config --trained_checkpoint_prefix training/model.ckpt-10856 --output_directory mac_n_cheese_inference_graph
    ```
 2. Your checkpoint files should be in the ```training``` directory. Just look for the one with the largest step (the largest number after the dash), and that's the one you want to use. Next, make sure the pipeline_config_path is set to whatever config file you chose, and then finally choose the name for the output directory, I went with mac_n_cheese_inference_graph
 
 3. You should have a new directory, ```mac_n_cheese_inference_graph```, inside it, we have new checkpoint data, a saved_model directory, and, most importantly, the ```forzen_inference_graph.pb``` file.
 
 4. Now, we're just going to use the tutorial code, edit it, and see how our model does on some testing images. We can copy some of my ```models/object_detection/images/test``` images into the ```models/object_detection/test_images``` directory, and renamed them to be ```image3.jpg, image4.jpg...etc```.
 
 5. Edit the tutorial with following code:
    ```python
    # What model to download.
    MODEL_NAME = 'mac_n_cheese_inference_graph'

    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = os.path.join('training', 'object-detection.pbtxt')

    NUM_CLASSES = 1
    ```
    
    ```python
    TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(3, 8) ]
    ```
    
  6. Run, see the result.
  
