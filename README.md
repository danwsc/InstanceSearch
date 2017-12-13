# InstanceSearch
Instance Search based on Tensorflow Faster-RCNN

The Tensorflow Faster-RCNN model refered to is available at
https://github.com/smallcorgi/Faster-RCNN_TF#installation-sufficient-for-the-demo.

For this to work, first get the above code working on your system.

The InstanceSearch code uses Python 2.7 with
- Tensorflow
- Nearpy

The Tensorflow version I used had gpu support.  I was not able to get the cpu only version working and stopped debugging this when gpu support resources were made available.

The Instance Search algorithm is developed from 
https://github.com/imatge-upc/retrieval-2016-deepvision

which is implemented using Caffe. This algorithm ranks (and reranks) search results based on calculating and sorting through the entire database (which in the case of the Paris dataset, comes to about 6300 images) per query image.  This project facilitates search using Locality Sensitive Hashing from Nearpy.

To run the code, follow the sequence in the Caffe implementation.
