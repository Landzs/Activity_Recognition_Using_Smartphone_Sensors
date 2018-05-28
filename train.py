# CODE RUNS!!!

# LOADING PACKAGES ETC
import random
import lasagne
import theano
import time
import warnings
warnings.filterwarnings('ignore', 'In the strict mode,.+')
import numpy as np
import pandas as pd
import theano.tensor as T
from sliding_window import sliding_window

# Hardcoded number of sensor channels employed in the OPPORTUNITY challenge
NB_SENSOR_CHANNELS = 0
# Hardcoded number of classes in the gesture recognition problem
NUM_CLASSES = 0
# Hardcoded length of the sliding window mechanism employed to segment the data
SLIDING_WINDOW_LENGTH = 0
# Length of the input sequence after convolutional operations
FINAL_SEQUENCE_LENGTH = 0
# Hardcoded step of the sliding window mechanism employed to segment the data
SLIDING_WINDOW_STEP = 0
# Batch Size
BATCH_SIZE = 0
# Number filters convolutional layers
NUM_FILTERS = 0
# Size filters convolutional layers
FILTER_SIZE = 0
# Number of unit in the long short-term recurrent layers
NUM_UNITS_LSTM = 0



def save_info(data, params, training=False):
    df = ''
    filename = ''
    penalty, learning_rate = params
    if training:
        tr_loss = data
        filename = 'Parameters/penalty' + penalty + '.learning_rate' + \
                   learning_rate + 'tr_loss.csv'
        df = pd.DataFrame({
            'tr loss': tr_loss,
        })
    else:
        val_loss, val_acc = data
        filename = 'Parameters/penalty' + penalty + '.learning_rate' + \
                   learning_rate + 'val_acc_and_val_loss.csv'
        df = pd.DataFrame({
            'val accuracy': val_acc,
            'val loss': val_loss,
        })

    return df.to_csv(filename)

def opp_sliding_window(data_x, data_y, ws, ss):
    data_x = sliding_window(data_x, (ws, data_x.shape[1]), (ss, 1))
    data_y = np.asarray([[i[-1]] for i in sliding_window(data_y, ws, ss)])
    return data_x.astype(np.float32), data_y.reshape(len(data_y)).astype(np.uint8)

    # This is identical to the test network

def build_cnn(input_var=None):
    net = {}
    net['input'] = lasagne.layers.InputLayer((BATCH_SIZE, 1, SLIDING_WINDOW_LENGTH, NB_SENSOR_CHANNELS), input_var=input_var)

    net['conv1/5x1'] = lasagne.layers.Conv2DLayer(net['input'], NUM_FILTERS, (FILTER_SIZE, 1))
    net['conv2/5x1'] = lasagne.layers.Conv2DLayer(net['conv1/5x1'], NUM_FILTERS, (FILTER_SIZE, 1))
    net['conv3/5x1'] = lasagne.layers.Conv2DLayer(net['conv2/5x1'], NUM_FILTERS, (FILTER_SIZE, 1))
    net['conv4/5x1'] = lasagne.layers.Conv2DLayer(net['conv3/5x1'], NUM_FILTERS, (FILTER_SIZE, 1))

    net['shuff'] = lasagne.layers.DimshuffleLayer(net['conv4/5x1'], (0, 2, 1, 3))

    net['lstm1'] = lasagne.layers.LSTMLayer(lasagne.layers.dropout(net['shuff'], p=.5), NUM_UNITS_LSTM)
    net['lstm2'] = lasagne.layers.LSTMLayer(lasagne.layers.dropout(net['lstm1'], p=.5), NUM_UNITS_LSTM)
    # In order to connect a recurrent layer to a dense layer, it is necessary to flatten the first two dimensions
    # to cause each time step of each sequence to be processed independently (see Lasagne docs for further information)
    net['shp1'] = lasagne.layers.ReshapeLayer(net['lstm2'], (-1, NUM_UNITS_LSTM))
    net['prob'] = lasagne.layers.DenseLayer(lasagne.layers.dropout(net['shp1'], p=.5), NUM_CLASSES, nonlinearity=lasagne.nonlinearities.softmax)
    # Tensors reshaped back to the original shape
    net['shp2'] = lasagne.layers.ReshapeLayer(net['prob'], (BATCH_SIZE, FINAL_SEQUENCE_LENGTH, NUM_CLASSES))
    # Last sample in the sequence is considered
    net['output'] = lasagne.layers.SliceLayer(net['shp2'], -1, 1)
    return net

# OLD MINI BATCH FUNCTION
def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

# NEW MINI BATCH FUNCTION
def custom_iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
        inputs = inputs[indices]
        targets = targets[indices]
    batches = {'X': dict(), 'y': dict()}
    for j in range(int(len(inputs) / batchsize)):
        j_start = j * batchsize
        j_end = (j + 1) * batchsize
        batches['X'][j] = inputs[j_start:j_end]
        batches['y'][j] = targets[j_start:j_end]
    return batches

def ConfusionMatrix(Y, pred_labels):
    confusion_matrix = np.zeros(shape=(NUM_CLASSES, NUM_CLASSES))
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            confusion_matrix[i, j] = len(
                np.where(pred_labels[np.where(Y == i)[0]] == j)[0])
    return confusion_matrix

def train(learning_rate, penalty, sensors, data, filters=64, test=False):

    global NB_SENSOR_CHANNELS, \
           NUM_CLASSES, \
           SLIDING_WINDOW_LENGTH, \
           FINAL_SEQUENCE_LENGTH, \
           SLIDING_WINDOW_STEP, \
           BATCH_SIZE, NUM_FILTERS, \
           FILTER_SIZE, \
           NUM_UNITS_LSTM

    X_train, y_train, X_val, y_val, X_test, y_test = data
    ### EPOCHS ###
    num_epochs = 25
    if test:
       num_epochs = 3
       sample_train_start = 263369
       sample_val_start = 10306
       sample_test_start = 2600

       sample_train_end = 407030
       sample_val_end = 85256
       sample_test_end = 24964

       X_train = X_train[sample_train_start:sample_train_end]
       y_train = y_train[sample_train_start:sample_train_end]
       X_val = X_val[sample_val_start:sample_val_end]
       y_val = y_val[sample_val_start:sample_val_end]
       X_test = X_test[sample_test_start:sample_test_end]
       y_test = y_test[sample_test_start:sample_test_end]

    # HARDCODING...

    # from pudb import set_trace; set_trace()

    # Hardcoded number of sensor channels employed in the OPPORTUNITY challenge
    NB_SENSOR_CHANNELS = sensors

    # Hardcoded number of classes in the gesture recognition problem
    NUM_CLASSES = 6

    # Hardcoded length of the sliding window mechanism employed to segment the data
    SLIDING_WINDOW_LENGTH = 24

    # Length of the input sequence after convolutional operations
    FINAL_SEQUENCE_LENGTH = 8

    # Hardcoded step of the sliding window mechanism employed to segment the data
    SLIDING_WINDOW_STEP = 12

    # Batch Size
    BATCH_SIZE = 100

    # Number filters convolutional layers
    NUM_FILTERS = filters

    # Size filters convolutional layers
    FILTER_SIZE = 5

    # Number of unit in the long short-term recurrent layers
    NUM_UNITS_LSTM = 128

    # BUILDING STUFF

    ############################### Main program ################################
    assert NB_SENSOR_CHANNELS == X_train.shape[1]

    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    # FURTHER DATA MANAGEMENT

    # train data
    print("Train with {0} instances in mini-batches of {1}".format(
        X_train.shape[0],
        BATCH_SIZE))
    # Sensor data is segmented using a sliding window mechanism
    X_train, y_train = opp_sliding_window(X_train, y_train,
                                          SLIDING_WINDOW_LENGTH,
                                          SLIDING_WINDOW_STEP)
    print(" ..after sliding window (testing): inputs {0}, targets {1}".format(
        X_train.shape, y_train.shape))
    # Data is reshaped since the input of the network is a 4 dimension tensor
    X_train = X_train.reshape(
        (-1, 1, SLIDING_WINDOW_LENGTH, NB_SENSOR_CHANNELS))
    print(" ..shape after reshaping: inputs {0}".format(X_train.shape))

    # validation data
    print(
        "Validate with {0} instances in mini-batches of {1}".format(
            X_val.shape[0],
            BATCH_SIZE))
    # Sensor data is segmented using a sliding window mechanism
    X_val, y_val = opp_sliding_window(X_val, y_val, SLIDING_WINDOW_LENGTH,
                                      SLIDING_WINDOW_STEP)
    print(" ..after sliding window (testing): inputs {0}, targets {1}".format(
        X_val.shape, y_val.shape))
    # Data is reshaped since the input of the network is a 4 dimension tensor
    X_val = X_val.reshape((-1, 1, SLIDING_WINDOW_LENGTH, NB_SENSOR_CHANNELS))
    print(" ..shape after reshaping: inputs {0}".format(X_train.shape))

    # test data
    print(
        "Test with {0} instances in mini-batches of {1}".format(X_test.shape[0],
                                                                BATCH_SIZE))
    # Sensor data is segmented using a sliding window mechanism
    X_test, y_test = opp_sliding_window(X_test, y_test, SLIDING_WINDOW_LENGTH,
                                        SLIDING_WINDOW_STEP)
    print(" ..after sliding window (testing): inputs {0}, targets {1}".format(
        X_test.shape, y_test.shape))
    # Data is reshaped since the input of the network is a 4 dimension tensor
    X_test = X_test.reshape((-1, 1, SLIDING_WINDOW_LENGTH, NB_SENSOR_CHANNELS))
    print(" ..shape after reshaping: inputs {0}".format(X_test.shape))

    # BUILDING NETWORK
    print("Building model and compiling functions...")
    net = build_cnn(input_var)

    # SPECIFYING TRAINING FUNCTIONS
    train_prediction_rnn = lasagne.layers.get_output(net['output'], input_var,
                                                     deterministic=False)
    train_loss_rnn = lasagne.objectives.categorical_crossentropy(
        train_prediction_rnn, target_var)
    train_loss_rnn = train_loss_rnn.mean()
    train_loss_rnn += penalty * lasagne.regularization.regularize_network_params(
        net['output'], lasagne.regularization.l2)
    params_rnn = lasagne.layers.get_all_params(net['output'], trainable=True)
    lr = learning_rate
    rho = 0.9
    updates_rnn = lasagne.updates.rmsprop(train_loss_rnn, params_rnn,
                                          learning_rate=lr, rho=rho)
    train_fn_rnn = theano.function([input_var, target_var], train_loss_rnn,
                                   updates=updates_rnn)
    # We could add some weight penalty as well here, see lasagne.regularization.

    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(net['output'],
                                                deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    pred_labels_fn = theano.function([input_var],
                                     [T.argmax(test_prediction, axis=1)])

    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    # TRAINING THE NETWORK
    tr_conf_matrix = dict()
    val_conf_matrix = dict()

    # keep track for plots
    validation_acc = []
    training_loss = []
    validation_loss = []

    print("Starting training...")
    # We iterate over epochs:
    for epoch in range(num_epochs):
        tr_conf_matrix[epoch] = np.zeros((NUM_CLASSES, NUM_CLASSES))
        val_conf_matrix[epoch] = np.zeros((NUM_CLASSES, NUM_CLASSES))
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        batches_training = custom_iterate_minibatches(X_train, y_train,
                                                      BATCH_SIZE, shuffle=True)
        for i in range(len(batches_training['X'])):
            train_err += train_fn_rnn(batches_training['X'][i],
                                      batches_training['y'][i])
            train_batches += 1
            tr_conf_matrix[epoch] += ConfusionMatrix(batches_training['y'][i],
                                                     pred_labels_fn(
                                                         batches_training['X'][
                                                             i])[
                                                         0])

        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        batches_validation = custom_iterate_minibatches(X_val, y_val,
                                                        BATCH_SIZE,
                                                        shuffle=False)
        for i in range(len(batches_validation['X'])):
            err, acc = val_fn(batches_validation['X'][i],
                              batches_validation['y'][i])
            val_err += err
            val_acc += acc
            val_batches += 1
            val_conf_matrix[epoch] += ConfusionMatrix(
                batches_validation['y'][i],
                pred_labels_fn(
                    batches_validation['X'][
                        i])[0])

        validation_acc.append(val_acc/val_batches * 100)
        validation_loss.append(val_err/ val_batches)
        training_loss.append(train_err / train_batches)

        # results for current epoch
        print("Epoch {} of {} took {:.3f}s".format(epoch + 1, num_epochs,
                                                   time.time() - start_time))
        print(
            "  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(
            val_acc / val_batches * 100))

        if val_acc / val_batches * 100 < 30:
            break


        save_as = 'Result/epoch' + str(epoch)
        save_as += '.penalty' + str(penalty) + '.learning_rate' + str(
            learning_rate)
        save_as += '.val_conf_matrix.csv'
        np.savetxt(save_as, val_conf_matrix[epoch], fmt='%10.0f')

        save_as = 'Result/epoch' + str(epoch)
        save_as += '.penalty' + str(penalty) + '.learning_rate' + str(
            learning_rate)
        save_as += '.tr_conf_matrix.csv'
        np.savetxt(save_as, tr_conf_matrix[epoch], fmt='%10.0f')

    # After training, we compute and print the test error:
    test_conf_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES))
    test_err = 0
    test_acc = 0
    test_batches = 0
    batches_test = custom_iterate_minibatches(X_test, y_test,
                                              BATCH_SIZE,
                                              shuffle=False)
    for i in range(len(batches_test['X'])):
        err, acc = val_fn(batches_test['X'][i], batches_test['y'][i])
        test_err += err
        test_acc += acc
        test_batches += 1
        test_conf_matrix += ConfusionMatrix(batches_test['y'][i], pred_labels_fn(batches_test['X'][i])[0])

    # save test conf matrix
    save_as = 'Result/test_acc' + str(test_acc / test_batches * 100)
    save_as += 'epoch' + str(epoch)
    save_as += 'penalty' + str(penalty) + 'learning_rate' + str(learning_rate)
    save_as += 'test_conf_matrix.csv'
    np.savetxt(save_as, test_conf_matrix, fmt='%10.0f')

    # save for plots
    save_info((validation_loss, validation_acc),(str(penalty), str(learning_rate)))
    save_info(training_loss, (str(penalty), str(learning_rate)), True)

    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(test_acc / test_batches * 100))

    # Dump the network weights to a file like this:
    name = 'Result/test_acc' + str(test_acc / test_batches * 100)
    name += 'penalty' + str(penalty) + 'learning_rate' + str(learning_rate) + 'model.npz'
    np.savez(name, *lasagne.layers.get_all_param_values(net['output']))


if __name__ == '__main__':
    random.seed(337)
    # LOAD FULL DATASET
    X_train = np.loadtxt('Project_Data/X_train.csv').astype(np.float32)
    y_train = np.loadtxt('Project_Data/Y_train.csv').astype(np.uint8)

    X_val = np.loadtxt('Project_Data/X_val.csv').astype(np.float32)
    y_val = np.loadtxt('Project_Data/Y_val.csv').astype(np.uint8)

    X_test = np.loadtxt('Project_Data/X_test.csv').astype(np.float32)
    y_test = np.loadtxt('Project_Data/Y_test.csv').astype(np.uint8)




    # train with suggested parameters
    learning = 0.0001
    penalty = 0.0001
    train(learning, penalty, 16, (X_train, y_train, X_val, y_val, X_test, y_test),64,True)

    # # parameter search 10 pairs of numbers
    # for learning, penalty in zip((random.uniform(0.01, 0.0001) for x in xrange(10)),
    #                            (random.uniform(0.001, 0.00001) for x in xrange(10))):
    #     print 'learning_rate:', learning, 'penalty', penalty
    #     train(learning, penalty, 16, (X_train, y_train, X_val, y_val, X_test, y_test))
    #
    # # train model with best parameters on full dataset
    # learning = 0.00104912881329
    # penalty = 0.000845993635181
    #
    # train(learning, penalty, 16, (X_train, y_train, X_val, y_val, X_test, y_test))
    #
    # # train model with 32 filters
    # train(learning, penalty, 16, (X_train, y_train, X_val, y_val, X_test, y_test), 32)
    #
    # # trim dataset of the personal attributes
    # X_train = np.loadtxt('Project_Data/X_train.csv').astype(np.float32)
    # X_train = np.delete(X_train, np.s_[12:X_train.shape[1]], axis=1)
    #
    # y_train = np.loadtxt('Project_Data/Y_train.csv').astype(np.uint8)
    #
    # X_val = np.loadtxt('Project_Data/X_val.csv').astype(np.float32)
    # X_val = np.delete(X_val, np.s_[12:X_val.shape[1]], axis=1)
    # y_val = np.loadtxt('Project_Data/Y_val.csv').astype(np.uint8)
    #
    # X_test = np.loadtxt('Project_Data/X_test.csv').astype(np.float32)
    # X_test = np.delete(X_test, np.s_[12:X_test.shape[1]], axis=1)
    # y_test = np.loadtxt('Project_Data/Y_test.csv').astype(np.uint8)
    #
    # # train with suggested parameters
    # learning = 0.0001
    # penalty = 0.0001
    # train(learning, penalty, 12, (X_train, y_train, X_val, y_val, X_test, y_test))
    #
    # # train with 32 filters
    # learning = 0.0001
    # penalty = 0.0001
    # train(learning, penalty, 12, (X_train, y_train, X_val, y_val, X_test, y_test), 32)
    #
    # # train model with best parameters
    # learning = 0.00104912881329
    # penalty = 0.000845993635181
    #
    # train(learning, penalty, 12, (X_train, y_train, X_val, y_val, X_test, y_test))
    #
    # # train with 32 filters
    # train(learning, penalty, 12, (X_train, y_train, X_val, y_val, X_test, y_test), 32)


