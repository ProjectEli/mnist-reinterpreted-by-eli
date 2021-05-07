# MNIST reinterpreted by Eli

* Basic code: https://www.tensorflow.org/overview
* plot method: https://www.tensorflow.org/tutorials/load_data/images
* visualization for mnist: https://webnautes.tistory.com/1232

## Import required module


```python
import tensorflow as tf
mnist = tf.keras.datasets.mnist
```

## Preprocess training data
since black images are set to RGB 0-255 integer, normalize it to 0-1 range for keras model


```python
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train,x_test = x_train/255.0, x_test/255.0
```

### set class names
keras sequential model will automatically sort class label with alphabetical order


```python
import numpy
class_names = numpy.unique(y_train)
print(class_names)
```

    [0 1 2 3 4 5 6 7 8 9]
    

### test data shape


```python
print(type(x_train))
print(x_train.shape)
print(x_train[1].shape)
print(y_train.shape)
```

    <class 'numpy.ndarray'>
    (60000, 28, 28)
    (28, 28)
    (60000,)
    

### train data plot


```python
import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
for i in range(9):
    ax = plt.subplot(3,3,i+1)
    plt.imshow(x_train[i],cmap='Greys')
    plt.title('true label: {}'.format(y_train[i]))
    plt.axis('off')
```


    
![png](doc/output_11_0.png)
    


## Setup keras model
each mnist image is consist of 28,28 size => pipeline 1  
pipeline 1 => make keras network with 128 units, and activation function as relu => pipeline2  
pipeline 2 => make 20% of nodes to 0 to prevent overfitting => pipeline 3  
pipeline 3 => make keras network with 10 nodes, activation function as softmax  

input size = 60000  
layer1 = 128 nodes  
output size = 10 (mnist number of label 0-9)
- [X] number of last layer should be same as the number of category unit  

category unit will be determined at future compile method: sparse_categorical_crossentropy


```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10,activation='softmax')
])
```

### summary of model
784+1(bias) * 128 = 100480  
128+1(bias) * 10 = 1290


```python
model.summary()
```

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    flatten (Flatten)            (None, 784)               0         
    _________________________________________________________________
    dense (Dense)                (None, 128)               100480    
    _________________________________________________________________
    dropout (Dropout)            (None, 128)               0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 10)                1290      
    =================================================================
    Total params: 101,770
    Trainable params: 101,770
    Non-trainable params: 0
    _________________________________________________________________
    

## Specify optimizer and a loss function
ref: https://keras.io/getting_started/intro_to_keras_for_engineers/#training-models-with-fit  
for loss function(great article!!):  https://gombru.github.io/2018/05/23/cross_entropy_loss/  


```python
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics = ['accuracy'])
```

## Start fitting process


```python
model.fit(x_train,y_train,epochs=10)
```

    Epoch 1/10
    1875/1875 [==============================] - ETA: 0s - loss: 0.4881 - accuracy: 0.8554 ETA: 1s - loss: 0.9678 - accuracy: 0. - ETA: 1s - loss: 0.8640 - accuracy:  - ETA: 1s - loss: 0.7544 - accuracy: 0.77 - ETA: 1s - loss: 0.7275 - accuracy: 0.78 - ETA: 0s - loss: 0.7019 - accuracy:  - ETA: 0s - loss: 0.6497 - accuracy: 0. - ETA: 0s - loss: 0.6187 - accuracy: 0.81 - ETA: 0s - loss: 0.6056 - accuracy: 0. - ETA: 0s - loss: 0.5862 -  - ETA: 0s - loss: 0.5180 - accuracy: 0.84 - ETA: 0s - loss: 0.5102 - accuracy: 0.84 - ETA: 0s - loss: 0.5027 - accuracy:  - 2s 895us/step - loss: 0.4813 - accuracy: 0.8574
    Epoch 2/10
    1875/1875 [==============================] - ETA: 0s - loss: 0.1508 - accuracy: 0.9556 ETA: 1s - loss: 0.1478 - accuracy:  - ETA: 1s - loss: 0.1441 - accuracy: 0. - ETA: 1s - loss: 0.1486 - accuracy: 0. - ETA: 1s - loss: 0.1513 - accuracy: 0. - ETA: 1s - loss: 0.1525 - accuracy:  - ETA: 0s - loss: 0.1534 - accuracy: 0.95 - ETA: 0s - loss: 0.1535 - accuracy: 0.95 - ETA: 0s - loss: 0.1535 - accuracy: 0. - ETA: 0s - loss: 0.1533 - accuracy: 0. - ETA: 0s - loss: 0.1531 - accuracy: 0.95 - ETA: 0s - loss: 0.1530 - accuracy: 0. - ETA: 0s - loss: 0.1527 - accuracy: 0.95 - ETA: 0s - loss: 0.1524 - accuracy - ETA: 0s - loss: 0.1515 - accuracy: 0.95 - ETA: 0s - loss: 0.1512 - accuracy: 0.95 - ETA: 0s - loss: 0.1510 - accuracy: 0. - 2s 835us/step - loss: 0.1506 - accuracy: 0.9557
    Epoch 3/10
    1875/1875 [==============================] - 2s 873us/step - loss: 0.1041 - accuracy: 0.96831s - loss: 0.0897 - accuracy: 0.97 - ETA: 1s - loss: 0.0941 - accuracy: 0. - ETA: 1s - loss: 0.0971 - accuracy: 0.97 - ETA: 1s - loss: 0.0976 - accuracy:  - ETA: 1s - loss: 0.0989 - accuracy: 0.96 - ETA: 1s - loss: 0.0994 - accuracy: 0. - ETA: 1s - loss: 0.1007 - accuracy: 0. - ETA: 0s - loss: 0.1016 - accuracy: 0.96 - ETA: 0s - loss: 0.1020 - accuracy: 0.96 - ETA: 0s - loss: 0.1025 - accuracy: 0.96 - ETA: 0s - loss: 0.1029 - accuracy: 0. - ETA: 0s - loss: 0.1035 - accuracy: 0.96 - ETA: 0s - loss: 0.1036 - accuracy: 0.96 - ETA: 0s - loss: 0.1037 - accuracy: 0.96 - ETA: 0s - loss: 0.1038 - accuracy:  - ETA: 0s - loss: 0.1040 - accuracy: 0. - ETA: 0s - loss: 0.1041 - accuracy: 0. - ETA: 0s - loss: 0.1041 - accuracy: 0.96 - ETA: 0s - loss: 0.1041 - accuracy
    Epoch 4/10
    1875/1875 [==============================] - 2s 825us/step - loss: 0.0844 - accuracy: 0.97421s - loss: 0.0700 - accuracy - ETA: 1s - loss: 0.0785 - accuracy: 0.97 - ETA: 1s - loss: 0.0797 - accuracy:  - ETA: 1s - loss: 0.0820 - accuracy: 0.97 - ETA: 0s - loss: 0.0825 - accuracy: 0.97 - ETA: 0s - loss: 0.0830 - accuracy: 0. - ETA: 0s - loss: 0.0835 - accuracy: 0. - ETA: 0s - loss: 0.0836 - accuracy: 0.97 - ETA: 0s - loss: 0.0837 - accuracy: 0. - ETA: 0s - loss: 0.0839 - accuracy: 0. - ETA: 0s - loss: 0.0840 - accuracy:  - ETA: 0s - loss: 0.0841 - accuracy: 0.97 - ETA: 0s - loss: 0.0841 - accuracy - ETA: 0s - loss: 0.0843 - accuracy: 0.97 - ETA: 0s - loss: 0.0844 - accuracy: 0.97
    Epoch 5/10
    1875/1875 [==============================] - 2s 870us/step - loss: 0.0698 - accuracy: 0.97711s - loss: 0.0633 - accuracy: 0.97 - ETA: 1s - loss: 0.0673 - accuracy:  - ETA: 1s - loss: 0.0682 - accuracy: 0.97 - ETA: 1s - loss: 0.0681 - accuracy: 0. - ETA: 1s - loss: 0.0678 - accuracy: 0. - ETA: 1s - loss: 0.0678 - accuracy:  - ETA: 0s - loss: 0.0680 - accuracy: 0. - ETA: 0s - loss: 0.0681 - accuracy: 0.97 - ETA: 0s - loss: 0.0681 - accuracy: 0.97 - ETA: 0s - loss: 0.0682 - accuracy: 0.97 - ETA: 0s - loss: 0.0683 - accuracy - ETA: 0s - loss: 0.0686 - accuracy: 0. - ETA: 0s - loss: 0.0689 - accuracy: 0.97 - ETA: 0s - loss: 0.0690 - accuracy: 0. - ETA: 0s - loss: 0.0693 - accuracy: 0.97 - ETA: 0s - loss: 0.0694 - accuracy: 0.97 - ETA: 0s - loss: 0.0695 - accuracy: 0.97 - ETA: 0s - loss: 0.0696 - accuracy: 0.97 - ETA: 0s - loss: 0.0697 - accuracy: 0.97 - ETA: 0s - loss: 0.0698 - accuracy: 0.97
    Epoch 6/10
    1875/1875 [==============================] - 2s 814us/step - loss: 0.0626 - accuracy: 0.98051s - loss: 0.0606 - accuracy:  - ETA: 1s - loss: 0.0610 - accuracy: 0.98 - ETA: 1s - loss: 0.0613 - accuracy: 0.98 - ETA: 0s - loss: 0.0615 - accuracy - ETA: 0s - loss: 0.0619 - accuracy: 0. - ETA: 0s - loss: 0.0620 - accuracy: 0.98 - ETA: 0s - loss: 0.0621 - accuracy: 0.98 - ETA: 0s - loss: 0.0621 - accuracy: 0.98 - ETA: 0s - loss: 0.0621 - accuracy - ETA: 0s - loss: 0.0624 - accuracy: 0.98 - ETA: 0s - loss: 0.0624 - accuracy: 0. - ETA: 0s - loss: 0.0625 - accuracy: 0.98 - ETA: 0s - loss: 0.0626 - accuracy: 0. - ETA: 0s - loss: 0.0626 - accuracy: 0.98
    Epoch 7/10
    1875/1875 [==============================] - 2s 843us/step - loss: 0.0546 - accuracy: 0.98181s - loss: 0.0545 - accuracy - ETA: 1s - loss: 0.0547 - accuracy: 0.98 - ETA: 1s - loss: 0.0546 - accuracy - ETA: 1s - loss: 0.0542 - accuracy: 0.98 - ETA: 1s - loss: 0.0542 - accuracy - ETA: 0s - loss: 0.0544 - accuracy: 0.98 - ETA: 0s - loss: 0.0544 - accuracy:  - ETA: 0s - loss: 0.0543 - accuracy: 0.98 - ETA: 0s - loss: 0.0543 - accuracy: 0.98 - ETA: 0s - loss: 0.0543 - accuracy: 0.98 - ETA: 0s - loss: 0.0543 - accuracy: 0. - ETA: 0s - loss: 0.0543 - accuracy: 0.98 - ETA: 0s - loss: 0.0543 - accuracy: 0.98 - ETA: 0s - loss: 0.0544 - accuracy:  - ETA: 0s - loss: 0.0545 - accuracy: 0.98 - ETA: 0s - loss: 0.0545 - accuracy: 0.
    Epoch 8/10
    1875/1875 [==============================] - 2s 840us/step - loss: 0.0473 - accuracy: 0.98481s - loss: 0.0498 - accu - ETA: 1s - loss: 0.0444 - accuracy: 0. - ETA: 1s - loss: 0.0446 - accuracy: 0. - ETA: 1s - loss: 0.0446 - accuracy - ETA: 0s - loss: 0.0452 - accuracy: 0.98 - ETA: 0s - loss: 0.0453 - accuracy: 0.98 - ETA: 0s - loss: 0.0454 - accuracy: 0. - ETA: 0s - loss: 0.0458 - accuracy: 0.98 - ETA: 0s - loss: 0.0460 - accuracy: 0.98 - ETA: 0s - loss: 0.0462 - accuracy:  - ETA: 0s - loss: 0.0467 - accuracy - ETA: 0s - loss: 0.0471 - accuracy: 0.98 - ETA: 0s - loss: 0.0472 - accuracy: 0.98 - ETA: 0s - loss: 0.0473 - accuracy: 0.98
    Epoch 9/10
    1875/1875 [==============================] - 2s 843us/step - loss: 0.0428 - accuracy: 0.98561s - loss: 0.0413 - accuracy: 0.98 - ETA: 1s - loss: 0.0416 - accuracy: 0.98 - ETA: 1s - loss: 0.0413 - accuracy:  - ETA: 1s - loss: 0.0400 - accuracy:  - ETA: 1s - loss: 0.0399 - accuracy: 0.98 - ETA: 0s - loss: 0.0399 - accuracy: 0. - ETA: 0s - loss: 0.0401 - accuracy: 0.98 - ETA: 0s - loss: 0.0403 - accuracy: 0. - ETA: 0s - loss: 0.0407 - accuracy:  - ETA: 0s - loss: 0.0413 - accuracy: 0. - ETA: 0s - loss: 0.0415 - accuracy: 0.98 - ETA: 0s - loss: 0.0416 - accuracy: 0.98 - ETA: 0s - loss: 0.0417 - accuracy: 0.98 - ETA: 0s - loss: 0.0418 - accuracy: 0.98 - ETA: 0s - loss: 0.0420 - accuracy: 0.98 - ETA: 0s - loss: 0.0420 - accuracy: 0. - ETA: 0s - loss: 0.0423 - accuracy: 0.98 - ETA: 0s - loss: 0.0424 - accuracy: 0.98 - ETA: 0s - loss: 0.0425 - accuracy: 0.98 - ETA: 0s - loss: 0.0427 - accuracy: 0.
    Epoch 10/10
    1875/1875 [==============================] - ETA: 0s - loss: 0.0389 - accuracy: 0.9870 ETA: 1s - loss: 0.0351 - accuracy: 0.98 - ETA: 1s - loss: 0.0355 - accuracy: 0.98 - ETA: 1s - loss: 0.0357 - accuracy: 0.98 - ETA: 1s - loss: 0.0359 - accuracy: 0. - ETA: 1s - loss: 0.0367 - accuracy: 0. - ETA: 1s - loss: 0.0372 - accuracy: 0. - ETA: 0s - loss: 0.0374 - accuracy: 0.98 - ETA: 0s - loss: 0.0374 - accuracy: 0.98 - ETA: 0s - loss: 0.0375 - accuracy: 0. - ETA: 0s - loss: 0.0378 - accuracy: 0.98 - ETA: 0s - loss: 0.0378 - accuracy: 0.98 - ETA: 0s - loss: 0.0379 - accuracy: 0. - ETA: 0s - loss: 0.0381 - accuracy: 0. - ETA: 0s - loss: 0.0382 - accuracy: 0.98 - ETA: 0s - loss: 0.0383 - accura - ETA: 0s - loss: 0.0387 - accuracy: 0.98 - ETA: 0s - loss: 0.0388 - accuracy: 0. - 2s 826us/step - loss: 0.0389 - accuracy: 0.9869
    




    <tensorflow.python.keras.callbacks.History at 0x2a2290989d0>



## Test data

### plot test data


```python
import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
for i in range(9):
    ax = plt.subplot(3,3,i+1)
    plt.imshow(x_test[i],cmap='Greys')
    plt.title('label: {}'.format(y_test[i]))
    plt.axis('off')
```


    
![png](doc/output_22_0.png)
    


### evaluate data


```python
model.evaluate(x_test,y_test)
```

    313/313 [==============================] - 0s 596us/step - loss: 0.0632 - accuracy: 0.9820
    




    [0.06319496780633926, 0.9819999933242798]



## Predict data
shuffle ref: https://www.tensorflow.org/tutorials/load_data/csv#%ED%9B%88%EB%A0%A8_%ED%8F%89%EA%B0%80_%EB%B0%8F_%EC%98%88%EC%B8%A1%ED%95%98%EA%B8%B0  
shuffle using zip: https://stackoverflow.com/questions/23289547/shuffle-two-list-at-once-with-same-order

### shuffle test set


```python
selectedindices = numpy.arange(x_test.shape[0])
x_sample,y_sample = x_test[selectedindices], y_test[selectedindices] # y sample is true label
```

### display sample size


```python
print(type(x_test))
print(x_sample[:9].shape)
```

    <class 'numpy.ndarray'>
    (9, 28, 28)
    

### prediction


```python
predictions = model.predict(x_sample[:9])
print(predictions.shape)
with numpy.printoptions(precision=2,suppress=True): # https://numpy.org/doc/stable/reference/generated/numpy.set_printoptions.html
    print(100*predictions)
```

    (9, 10)
    [[  0.     0.     0.     0.     0.     0.     0.   100.     0.     0.  ]
     [  0.     0.01  99.99   0.     0.     0.     0.     0.     0.     0.  ]
     [  0.    99.99   0.     0.     0.     0.     0.     0.01   0.     0.  ]
     [ 99.98   0.     0.     0.     0.     0.01   0.     0.01   0.     0.  ]
     [  0.     0.     0.     0.    99.98   0.     0.     0.01   0.     0.01]
     [  0.    99.94   0.     0.     0.     0.     0.     0.06   0.     0.  ]
     [  0.     0.     0.     0.    99.99   0.     0.     0.     0.01   0.  ]
     [  0.     0.     0.     0.01   0.07   0.     0.     0.     0.    99.91]
     [  0.     0.     0.     0.     0.    60.05  39.94   0.     0.02   0.  ]]
    

### prediction of most probable class names


```python
print(y_sample[:9])
```

    [7 2 1 0 4 1 4 9 5]
    

#### display most probable class names using argmax (`model.predict_classes()` is deprecated)


```python
prediction_classes_argmax = predictions.argmax(axis=-1)
print(prediction_classes_argmax)
```

    [7 2 1 0 4 1 4 9 5]
    

### prediction with full probability map

#### define some functions to plot data
ref: https://www.tensorflow.org/tutorials/keras/classification?hl=ko#%EC%98%88%EC%B8%A1_%EB%A7%8C%EB%93%A4%EA%B8%B0


```python
def plot_image(predictions, true_label_index, img):
    # predictions,true_label,img = predictions_array[i],true_label_array[i],img_array[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img,cmap='Greys')
    
    predicted_label_index = numpy.argmax(predictions) # to select most probable label index
    if predicted_label_index == true_label_index:
        color = 'blue'
    else:
        color = 'red'
    
    plt.xlabel('pred:{} {:2.0f}% (true:{})'.format(class_names[predicted_label_index], 100*numpy.max(predictions), class_names[true_label_index]), color=color)
    
def plot_value_array(predictions, true_label_index):
    plt.grid(False)
    plt.xticks(class_names)
    plt.yticks([])
    
    barplot = plt.bar(range(len(class_names)), predictions, color='#777777')
    plt.ylim([0,1])
    
    predicted_label_index = numpy.argmax(predictions)
    barplot[predicted_label_index].set_color('red')
    barplot[true_label_index].set_color('blue')
```

#### display all probability


```python
def plot_Nfigs(num_cols,num_rows,x_test,y_test,predictions):
    num_images = num_cols * num_rows
    plt.figure(figsize=(2*2*num_cols,2*num_rows)) # first 2 is figsize, second 2 is two plots
    for i in range(num_images):
        plt.subplot(num_rows,2*num_cols,2*i+1)
        plot_image(predictions[i],y_test[i],x_test[i])
        plt.subplot(num_rows,2*num_cols,2*i+2)
        plot_value_array(predictions[i],y_test[i])
plot_Nfigs(3,3,x_sample,y_sample,predictions)
```


    
![png](doc/output_40_0.png)
    


### Display wrong predictions

#### define wrong result finder function


```python
def wrong_results(x_test,y_test):
    x_test_wrong = []
    y_test_wrong = []
    predictions_wrong = []
    correct_results = 0
    
    predictions = model.predict(x_test)
    prediction_classes_argmax = predictions.argmax(axis=-1) # predicted class_index
    
    for i in range(y_test.shape[0]): #length of samples
        if y_test[i] != prediction_classes_argmax[i]:
            x_test_wrong.append(x_test[i])
            y_test_wrong.append(y_test[i])
            predictions_wrong.append(predictions[i])
        else:
            correct_results += 1
    x_test_wrong = numpy.array(x_test_wrong)
    y_test_wrong = numpy.array(y_test_wrong)
    predicitions_wrong = numpy.array(predictions_wrong)
    return x_test_wrong, y_test_wrong, predictions_wrong, correct_results
```

#### plot wrong results


```python
x_wrong, y_wrong, predictions_wrong, correct_results=wrong_results(x_test,y_test)
print(y_test.shape[0])
print(correct_results)
print(y_wrong.shape[0])

N = min(9, y_wrong.shape[0])
print(N)
if N==9:
    plot_Nfigs(3,3,x_wrong,y_wrong,predictions_wrong)
```

    10000
    9820
    180
    9
    


    
![png](doc/output_45_1.png)
    

