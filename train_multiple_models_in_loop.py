import os
import dlib
from pytictoc import TicToc
timer = TicToc()

# Generate all the 25 C values needed for optimization
C_values = []
for i in range(10, 0, -1):
    C_values.append(2**(-i))
for i in range(15):
    C_values.append(2**i)


# prepare files
head_folder = r"E:\images_folder"
training_xml_path = os.path.join(head_folder, "annotation.xml")
model_name_base = "trained_model"
model_name_ext = ".svm"

# define training options
options = dlib.simple_object_detector_training_options()
options.add_left_right_image_flips = True
# Tell the code how many CPU cores your computer has for the fastest training.
options.num_threads = 8
options.be_verbose = True
options.epsilon = 0.01
options.detection_window_size = 80*80

# Optimize the model by finding the best C value by
# iterating through 25 different C values pre-generated
for i in range(len(C_values)):
    timer.tic()  # start time

    # set C value
    options.C = C_values[i]

    # concatenate model file name
    model_name = model_name_base + str(i+1) + model_name_ext
    model_name = os.path.join(model_name)
    print(model_name)
    # training
    dlib.train_simple_object_detector(training_xml_path, model_name, options)

    timer.toc()  # end time

    # write accuracy to file
    f = open("Accuracy_Report.txt", "a+")
    f.write("\n\n")  # Print blank line to create gap from previous output
    f.write("==============================\n")
    f.write("Model: {}".format(model_name))
    f.write("\n")
    f.write("Training with C: {}".format(C_values[i]))
    f.write("\n")

    # write training duration
    if timer.elapsed >= 60:
        if timer.elapsed >= 360:
            t = timer.elapsed / 60 / 60
            m = (t % 1) * 60
            f.write("Duration of training: %d hour(s), %d minute(s)" % (t, m))
        else:
            t = timer.elapsed / 60
            f.write("Duration of training: %.2f" % t)
    else:
        f.write("Duration of training: %.2f seconds" % timer.elapsed)

    f.write("\n")
    f.write("Training accuracy: {}".format(
        dlib.test_simple_object_detector(training_xml_path, model_name)))
    f.close()
