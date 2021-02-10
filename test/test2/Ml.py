import tensorflow as tf
import numpy as np
import random
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))

x_data = []
y_data = []
x_test = []
y_test = []

x_data1 = [513, 513, 513, 513, 513]
x_data2 = [1000, 1000, 1000, 1000, 1000]
x_data3 = [1000, 525, 1000, 525, 1000]
x_data4 = [1020, 500, 700, 800, 900]
x_data5 = [1000, 900, 800, 700, 600]

y_data1 = 0
y_data2 = 1
y_data3 = 2
y_data4 = 3
y_data5 = 4


# 학습데이터 설정
def setData(add_x, add_y):
    x_data.append(add_x)
    for i in range(99):
        x_data.append([add_x[k] + add_x[k] * random.uniform(-0.1, 0.1) for k in range(5)])
    for i in range(100):
        one_hot = tf.one_hot(add_y, depth=5)
        one_hot = (tf.cast(one_hot, tf.float32)).numpy()
        one_hot = list(one_hot)
        y_data.append(one_hot)


setData(x_data1, y_data1)
setData(x_data2, y_data2)
setData(x_data3, y_data3)
setData(x_data4, y_data4)
setData(x_data5, y_data5)

scaler.fit(x_data)
x_data = scaler.transform(x_data)


# 테스트 데이터 설정
def setTest(add_x, add_y):
    x_test.append(add_x)
    for i in range(99):
        x_test.append([add_x[k] + add_x[k] * random.uniform(-0.15, 0.15) for k in range(5)])
    for i in range(100):
        one_hot = tf.one_hot(add_y, depth=5)
        one_hot = (tf.cast(one_hot, tf.float32)).numpy()
        one_hot = list(one_hot)
        y_test.append(one_hot)


setTest(x_data1, y_data1)
setTest(x_data2, y_data2)
setTest(x_data3, y_data3)
setTest(x_data4, y_data4)
setTest(x_data5, y_data5)

scaler.fit(x_test)
x_test = scaler.transform(x_test)

############################

dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data)).batch(len(x_data))
dataset.element_spec

initializer = tf.initializers.he_uniform()

W1 = tf.Variable(initializer(shape=([5, 10])))
b1 = tf.Variable(tf.random.normal([10]), name='bias1')

W2 = tf.Variable(initializer(shape=([10, 10])))
b2 = tf.Variable(tf.random.normal([10]), name='bias2')

W3 = tf.Variable(initializer(shape=([10, 10])))
b3 = tf.Variable(tf.random.normal([10]), name='bias3')

W = tf.Variable(initializer(shape=([10, 5])))
b = tf.Variable(tf.random.normal([5]), name='bias4')


def preprocess_data(features, labels):
    features = tf.cast(features, tf.float32)
    labels = tf.cast(labels, tf.float32)
    return features, labels


def neural_net(features):
    layer1 = tf.nn.relu(tf.matmul(features, W1) + b1)
    layer2 = tf.nn.relu(tf.matmul(layer1, W2) + b2)
    layer3 = tf.nn.relu(tf.matmul(layer2, W3) + b3)
    hypothesis = tf.nn.softmax(tf.matmul(layer3, W) + b)
    return hypothesis


def loss_fn(hypothesis, features, labels):
    cost = -tf.reduce_mean(labels * tf.math.log(hypothesis) + (1 - labels) * tf.math.log(1 - hypothesis))
    return cost


optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)


def accuracy_fn(hypothesis, labels):
    prediction = tf.argmax(hypothesis, 1)
    is_correct = tf.equal(prediction, tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
    return accuracy


@tf.function
def grad(features, labels):
    with tf.GradientTape() as tape:
        loss_value = loss_fn(neural_net(features), features, labels)
    return tape.gradient(loss_value, [W1, W2, W3, W, b1, b2, b3, b])


def forward(x):
    scaler.fit(x)
    x = scaler.transform(x)
    x = tf.cast(x, tf.float32)
    x = neural_net(x)
    return tf.argmax(x, 1)


EPOCHS = 1000
for step in range(EPOCHS + 1):
    for features, labels in iter(dataset):
        features, labels = preprocess_data(features, labels)
        hypothesis = neural_net(features)
        grads = grad(features, labels)
        optimizer.apply_gradients(grads_and_vars=zip(grads, [W1, W2, W3, W, b1, b2, b3, b]))
    if step % 100 == 0:
        print("Iter: {}, Loss: {:.4f}".format(step, loss_fn(neural_net(features), features, labels)))
        #tf.saved_model.save(neural_net(features),'/home/inlabws/SY_TEST/my_test_model')
        tf.saved_model.save(optimizer,'./my_test_model')

# 테스트 데이터로 테스트

x_test, y_test = preprocess_data(x_test, y_test)
test_acc = accuracy_fn(neural_net(x_test), y_test)
print("Testset Accuracy: {:.4f}".format(test_acc))

# 데이터 넣어보기

input = [[513, 513, 513, 513, 513],
         [1000, 1000, 1000, 1000, 1000],
         [1000, 520, 1000, 525, 1000],
         [1020, 500, 700, 800, 900],
         [1000, 900, 800, 700, 600]]

output = forward(input)
print(output)





