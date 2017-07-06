def transform_images(data):
    # load data
    (X_train, y_train) = #mnist.load_data()

    # reshape to be [samples][pixels][width][height]
    X_train = X_train.reshape(X_train.shape[0], 1, 48, 48).astype('float32')
    X_test = X_test.reshape(X_test.shape[0], 1, 48, 48).astype('float32')

    # normalize inputs from 0-255 to 0-1
    X_train = X_train / 255
    X_test = X_test / 255

    # one hot encode outputs
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    num_classes = y_test.shape[1]
