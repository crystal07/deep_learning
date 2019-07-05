import tensorflow as tf
import imageio
import numpy as np
import os
from datetime import datetime

CROP_SIZE = 32
DOWN_SIZE = 16
CHANNEL = 3
BATCH_SIZE = 128

# TRAINING_FOLDER = "./91"
TRAINING_FOLDER = "./291"
TEST_FOLDER = "./Set5"

image_list = os.listdir(TRAINING_FOLDER)
gray_image_list = []
test_gray_image_list = []
image_shape = []
for img in image_list :
    try :
        image = imageio.imread(os.path.join(TRAINING_FOLDER, img))
        shape = image.shape
        image_shape.append(shape)
        image = tf.image.rgb_to_grayscale(image)

        downsize_image = tf.image.resize_images(image, [int(shape[0]/2), int(shape[1]/2)])
        restored_image = tf.image.resize_images(downsize_image, [shape[0], shape[1]])

        # restored_image = tf.Session().run(restored_image) / 255
        # image = tf.Session().run(image) / 255
        restored_image = tf.Session().run(restored_image / 255)
        image = tf.Session().run(image / 255)

        gray_image_list.append(restored_image)
        test_gray_image_list.append(image)

    except Exception as e :
        print(e)

image_list = os.listdir(TEST_FOLDER)
test_training_set = []
test_image_shape = []
test_test_set = []
test_file_name = []
for img in image_list :
    try :
        image = imageio.imread(os.path.join(TEST_FOLDER, img))
        shape = image.shape
        test_image_shape.append(shape)
        test_file_name.append(img)
        image = tf.image.rgb_to_grayscale(image)

        downsize_image = tf.image.resize_images(image, [int(shape[0]/2), int(shape[1]/2)])
        restored_image = tf.image.resize_images(downsize_image, [shape[0], shape[1]])

        # restored_image = tf.Session().run(restored_image) / 255
        # image = tf.Session().run(restored_image) / 255
        restored_image = tf.Session().run(restored_image / 255)
        image = tf.Session().run(image / 255)

        test_training_set.append(restored_image.tolist())
        test_test_set.append(image.tolist())

    except Exception as e :
        print(e)

X = tf.placeholder(tf.float32, [None, None, None, 1])
Y = tf.placeholder(tf.float32, [None, None, None, 1])

W1 = tf.get_variable("W1", shape=[3,3,1,64], initializer=tf.contrib.layers.xavier_initializer())
W2 = tf.get_variable("W2", shape=[3,3,64,32], initializer=tf.contrib.layers.xavier_initializer())
W3 = tf.get_variable("W3", shape=[3,3,32,1], initializer=tf.contrib.layers.xavier_initializer())

b1 = tf.Variable(tf.zeros([64]))
b2 = tf.Variable(tf.zeros([32]))
b3 = tf.Variable(tf.zeros([1]))

tf.summary.histogram("W1", W1)
tf.summary.histogram("W2", W2)
tf.summary.histogram("W3", W3)

# gray_image_list = tf.cast(gray_image_list, tf.float32)
# test_gray_image_list = tf.cast(test_gray_image_list, tf.float32)

layer1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding="SAME")
layer1 = tf.nn.relu(layer1 + b1)
layer2 = tf.nn.conv2d(layer1, W2, strides=[1, 1, 1, 1], padding="SAME")
layer2 = tf.nn.relu(layer2 + b2)
layer3 = tf.nn.conv2d(layer2, W3, strides=[1, 1, 1, 1], padding="SAME")
hypothesis = layer3 + b3

with tf.name_scope("Cost") :
    # cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
    # cost = tf.reduce_mean(tf.square(hypothesis - Y))
    cost = tf.sqrt(tf.reduce_sum(tf.square(hypothesis - Y)))
    tf.summary.scalar("Cost", cost)

with tf.name_scope("Train") :
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.9, beta2=0.999, epsilon=1e-08).minimize(cost)

# accuracy = tf.reduce_mean(tf.cast(tf.equal(hypothesis, Y), dtype=tf.float32))
accuracy = tf.reduce_mean(tf.image.psnr(hypothesis, Y, max_val=1.0))
tf.summary.scalar("Accuracy", accuracy)

with tf.Session() as sess:
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter("./logs/srcnn_333_nor")
    writer.add_graph(sess.graph)  # Show the graph
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        # print("start step", datetime.now())
        x_data = []
        y_data = []
        for idx_ in range(len(gray_image_list)):
            shape = image_shape[idx_]
            widx, hidx = [np.random.randint(shape[0] - CROP_SIZE - 1, size=1)[0],
                          np.random.randint(shape[1] - CROP_SIZE - 1, size=1)[0]]
            cropped_image = []
            test_cropped_image = []
            for idx in range(widx, widx + CROP_SIZE):
                cropped_image.append(gray_image_list[idx_][idx][hidx:hidx + CROP_SIZE])
                test_cropped_image.append(test_gray_image_list[idx_][idx][hidx:hidx + CROP_SIZE])
            # print(cropped_image)
            x_data.append(cropped_image)
            y_data.append(test_cropped_image)

        index = 0
        while index < len(x_data):
            # print(step, index, index+BATCH_SIZE)
            _, summary, cost_val = sess.run([optimizer, merged_summary, cost],
                                            feed_dict={X: x_data[index:index + BATCH_SIZE],
                                                       Y: y_data[index:index + BATCH_SIZE]})
            index = index + BATCH_SIZE
        writer.add_summary(summary, global_step=step)
        if step % 200 == 0:
            print(step, cost_val)
        # print("end step", datetime.now())
    x_data = []
    y_data = []
    for idx_ in range(len(test_training_set)):
        shape = test_image_shape[idx_]
        widx, hidx = [np.random.randint(shape[0] - CROP_SIZE - 1, size=1)[0],
                      np.random.randint(shape[1] - CROP_SIZE - 1, size=1)[0]]
        cropped_image = []
        test_cropped_image = []
        for idx in range(widx, widx + CROP_SIZE):
            cropped_image.append(test_training_set[idx_][idx][hidx:hidx + CROP_SIZE])
            test_cropped_image.append(test_test_set[idx_][idx][hidx:hidx + CROP_SIZE])
        x_data.append(cropped_image)
        y_data.append(test_cropped_image)

    # Accuracy report
    h, a, c = sess.run(
        [hypothesis, accuracy, cost], feed_dict={X: x_data, Y: y_data}
    )
    #     print(x_data, y_data)

    for idx in range(len(test_file_name)):
        imageio.imwrite("result/" + "train" + test_file_name[idx], np.array(test_training_set[idx]))
        imageio.imwrite("result/" + "test" + test_file_name[idx], np.array(test_test_set[idx]))
        imageio.imwrite("result/" + "result" + test_file_name[idx],
                        hypothesis.eval(({X: [test_training_set[idx]], Y: [test_test_set[idx]]}))[0])
    #     for idx in range(len(test_file_name)):
    #         imageio.imwrite("test", hypothesis.eval(({X: [test_training_set[idx]], Y: [test_test_set[idx]]})))
    print("Accuracy {}\nCost {}\n".format(a, c))
    # print("\nHypothesis:\n{} \nAccuracy:\n{}".format(h, a))