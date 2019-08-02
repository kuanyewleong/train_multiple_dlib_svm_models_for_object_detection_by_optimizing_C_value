# Train multiple Dlib SVM models for object detection by optimizing C value

To get the best Dlib SVM model for object detection, apart from preprocess the data appropriately, one needs to tune (optimize) the C parameter. In this Python script, I demonstrated that the training process can be automated to train several models with different C values, and record their accuracy right after a model is trained. The C values are generated based on some literature that suggested to use 2^(-10) to 2^14. You may observe some of these values and adjust accordingly the range. 

PS: The image files and the annotation file are not provided. You may alter those files to your versions.
