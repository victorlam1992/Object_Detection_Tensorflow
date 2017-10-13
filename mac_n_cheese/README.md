## Mac and Cheese Detection (by sentdex, pythonprogramming.net)
we may check his tutorial at:

https://www.youtube.com/watch?v=COlbP62-B-U

https://pythonprogramming.net/custom-objects-tracking-tensorflow-object-detection-api-tutorial/

## Directory:
- data: 
	 - label of train & test data (CSV), tfrecord of train & test data

- image: 
	- raw image & XML of train & test data

- ssd_mobilenet_v1_coco_11_06_2017: 
	- pre-trained model

- training: 
	- config of pre-trained model, object label for mac_n_cheese

## File:
- labelImg_windows_v1.5.1.zip:
	- To create XML label for train and test data

- xml_to_csv.py: 
	- transform the XML label to CSV format
	
- generate_tfrecord.py:
	- transform the CSV to tfrecord format

