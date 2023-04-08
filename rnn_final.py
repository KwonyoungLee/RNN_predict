import pandas_datareader as pd_reader
import pandas as pd
import matplotlib.pyplot as plt
import sys
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import math
from sklearn.metrics import mean_squared_error


def sigmoid_forward(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_backward(x, ds):
    return x * (1 - x) * ds


def relu_forward(x):
    return np.maximum(x, 0)


def relu_backward(x, ds):
    return np.clip(x, 0, 1) * ds


def tanh_forward(x):
    return np.tanh(x)


def tanh_backward(x, ds):
    return (1.0 - np.square(x)) * ds


def calc_loss(X, Y):
    loss = 0.0

    for i in range(Y.shape[0]):
        x, y = X[i], Y[i]
        prev_s = np.zeros((hidden_dim, 1))
        for t in range(T):
            new_input = np.zeros(x.shape)
            new_input[t] = x[t]
            mulu = np.zeros((hidden_dim, 1))
            for k in range(new_input.shape[1]):
                new_dim = np.expand_dims(new_input[:, k], axis=1)
                mulu += np.dot(U[:, :], new_dim)
            mulw = np.dot(W, prev_s)
            add = mulw + mulu + bh
            if activation_function == 'sigmoid':
                s = sigmoid_forward(add)
            if activation_function == 'tanh':
                s = tanh_forward(add)
            if activation_function == 'relu':
                s = relu_forward(add)
            mulv = np.dot(V, s) + by
            prev_s = s
        loss_per_record = (y - mulv) ** 2 / 2
        loss += loss_per_record
    loss = loss / float(y.shape[0])
    return loss


tickers = [sys.argv[1]]
mmc = [sys.argv[2]]
hdmm = [sys.argv[3]]
ac_fun = [sys.argv[4]]


for tii in range(len(tickers)):
    for acf in range(len(ac_fun)):
        table = np.empty((9, 3), dtype=object)
        c = 0
        for hdm in range(len(hdmm)):
            for mmclip in range(len(mmc)):
                figure1, axis = plt.subplots(2,2, figsize=(24, 14))
                ticker = tickers[tii]
                # set random seed
                np.random.seed(1234)
                data_source = 'yahoo'
                s_date = '2020-05-15'
                e_date = '2022-05-15'
                df = pd_reader.DataReader(ticker, data_source, s_date, e_date)
                df = df.dropna()
                data = df
                dataset = data.values
                print(dataset.shape)

                min_clip = -1 * float(mmc[mmclip])
                max_clip = float(mmc[mmclip])

                #activation function
                activation_function = ac_fun[acf]

                # normalize
                sc = MinMaxScaler(feature_range=(0, 1))
                sc_data = sc.fit_transform(dataset)

                tbptt = 4

                #make training set as 80percent of data
                training_data_len = math.ceil(len(dataset) * 0.8)
                # learning rate
                learning_rate = 1e-3
                # number of iteration
                nepoch = 10
                # length of sequence
                T = 50
                # number of neuron in hidden layer
                hidden_dim = int(hdmm[hdm])
                output_dim = 1
                train_data = sc_data[0:training_data_len, :]


                print('Ticker: ' + str(ticker) + ' min and max: ' + str(min_clip) + ' ' + str(
                    max_clip) + ' hidden dim: ' + str(hidden_dim) + ' activation function: ' + str(activation_function))

                X = []
                Y = []

                for i in range(T, len(train_data)):
                    X.append(train_data[i - T:i, 3])
                    Y.append(train_data[i, 3])
                X = np.array(X)
                X = np.reshape(X, (X.shape[0], X.shape[1], 1))

                Y = np.array(Y)
                Y = np.expand_dims(Y, axis=1)
                #print('Y,', Y.shape)
                #print('X,', X.shape)

                test_data = sc_data[training_data_len - T:, :]

                X_val = []
                Y_val = []

                for i in range(T, len(test_data)):
                    X_val.append(test_data[i - T:i, 3])
                    Y_val.append(test_data[i, 3])

                X_val = np.array(X_val)
                X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], 1))

                Y_val = np.array(Y_val)
                Y_val = np.expand_dims(Y_val, axis=1)
                #print('Y_val,', Y_val.shape)

                #setting the initial weights
                #input -> hidden weight
                U = np.random.uniform(0, 1, (hidden_dim, T))
                #hidden -> hidden weight
                W = np.random.uniform(0, 1, (hidden_dim, hidden_dim))
                #hidden -> output weight
                V = np.random.uniform(0, 1, (output_dim, hidden_dim))

                #biases weight
                bh = np.zeros((hidden_dim, 1))
                by = np.zeros((output_dim, 1))

                loss_arr = []
                val_loss_arr = []
                # start iteration
                for epoch in range(nepoch):
                    # check loss on train
                    train_loss = calc_loss(X, Y)
                    # check loss on val
                    val_loss = calc_loss(X_val, Y_val)

                    loss_arr.append(train_loss[0, 0])
                    val_loss_arr.append(val_loss[0, 0])

                    print('Epoch: ', epoch + 1, ', Loss: ', train_loss, ', Val Loss: ', val_loss)

                    # train model
                    for i in range(Y.shape[0]):
                        x, y = X[i], Y[i]

                        layers = []
                        prev_s = np.zeros((hidden_dim, 1))
                        dU = np.zeros(U.shape)
                        dV = np.zeros(V.shape)
                        dW = np.zeros(W.shape)

                        dU_t = np.zeros(U.shape)
                        dV_t = np.zeros(V.shape)
                        dW_t = np.zeros(W.shape)

                        dU_i = np.zeros(U.shape)
                        dW_i = np.zeros(W.shape)

                        d_bh = np.zeros(bh.shape)

                        # forward pass
                        for t in range(T):
                            new_input = np.zeros(x.shape)
                            new_input[t] = x[t]
                            mulu = np.zeros((hidden_dim, 1))
                            for k in range(new_input.shape[1]):
                                new_dim = np.expand_dims(new_input[:, k], axis=1)
                                mulu += np.dot(U[:, :], new_dim)
                            mulw = np.dot(W, prev_s)
                            add = mulw + mulu + bh
                            if activation_function == 'sigmoid':
                                s = sigmoid_forward(add)
                            if activation_function == 'tanh':
                                s = tanh_forward(add)
                            if activation_function == 'relu':
                                s = relu_forward(add)
                            mulv = np.dot(V, s) + by
                            layers.append({'s': s, 'prev_s': prev_s})
                            prev_s = s

                        dmulv = (mulv - y)

                        # backward pass
                        for t in range(T):
                            d_by = dmulv
                            dV_t = np.dot(dmulv, np.transpose(layers[t]['s']))
                            dsv = np.dot(np.transpose(V), dmulv)
                            ds = dsv
                            if activation_function == 'sigmoid':
                                dadd = sigmoid_backward(add, ds)
                            if activation_function == 'tanh':
                                dadd = tanh_backward(add, ds)
                            if activation_function == 'relu':
                                dadd = relu_backward(add, ds)
                            d_bh += dadd
                            dmulw = dadd * np.ones_like(ds)
                            dprev_s = np.dot(np.transpose(W), dmulw)

                            for i in range(t - 1, max(-1, t - tbptt - 1), -1):
                                ds = dsv + dprev_s

                                if activation_function == 'sigmoid':
                                    dadd = sigmoid_backward(add, ds)
                                if activation_function == 'tanh':
                                    dadd = tanh_backward(add, ds)
                                if activation_function == 'relu':
                                    dadd = relu_backward(add, ds)
                                dmulw = dadd * np.ones_like(ds)
                                dprev_s = np.dot(np.transpose(W), dmulw)

                                dW_i = np.dot(W, layers[t]['prev_s'])

                                new_input = np.zeros(x.shape)
                                new_input[t] = x[t]
                                for k in range(new_input.shape[1]):
                                    new_dim = np.expand_dims(new_input[:, k], axis=1)
                                    dU_i += np.dot(U[:, :], new_dim)
                                dU_t += dU_i
                                dW_t += dW_i
                            dV += dV_t
                            dU += dU_t
                            dW += dW_t

                        # update the weight matrix
                        for d in [dV, dW, dU, d_bh, d_by]:
                            np.clip(d, min_clip, max_clip, out=d)
                        V -= learning_rate * dV
                        U -= learning_rate * dU
                        W -= learning_rate * dW
                        bh -= learning_rate * d_bh
                        by -= learning_rate * d_by

                # train
                preds = []
                for i in range(Y.shape[0]):
                    x, y = X[i], Y[i]
                    prev_s = np.zeros((hidden_dim, 1))
                    # Forward pass
                    for t in range(T):
                        mulu = np.zeros((hidden_dim, 1))
                        # # print(x.shape[1])
                        for k in range(x.shape[1]):
                            new_dim = np.expand_dims(x[:, k], axis=1)
                            mulu += np.dot(U[:, :], new_dim)
                            # print(U[:,:,k].shape)
                        mulw = np.dot(W, prev_s)
                        add = mulw + mulu + bh
                        if activation_function == 'sigmoid':
                            s = sigmoid_forward(add)
                        if activation_function == 'tanh':
                            s = tanh_forward(add)
                        if activation_function == 'relu':
                            s = relu_forward(add)
                        mulv = np.dot(V, s) + by
                        prev_s = s
                    preds.append(mulv)

                preds = np.array(preds)
                preds = np.reshape(preds, (len(preds), 1))
                preds_like = np.zeros(shape=(len(preds), 6))
                preds_like[:, 0] = preds[:, 0]
                preds = sc.inverse_transform(preds_like)[:, 0]

                train = data[T:training_data_len]
                train['Predictions'] = preds
                # print(valid)
                rmse = np.sqrt(mean_squared_error(train['Predictions'], train['Close']))
                print('training rmse = ', rmse)

                #axis[0,0].set_figure(figsize=(12, 6))
                axis[0,0].set_title('Training Ticker model: {}, rmse= {}, clip max= {}, h_dim= {}, {}'.format(ticker, rmse, max_clip, hidden_dim, activation_function))
                axis[0,0].set_xlabel('Date')
                axis[0,0].set_ylabel('Close Price')
                # plt.plot(train['Close'])
                axis[0,0].plot(train[['Close', 'Predictions']])
                axis[0,0].legend(['Training data', 'Predictions'], loc='lower left')
                #plt.savefig('Training {} rmse= {} Cmax= {} Hdim= {} {}.jpg'.format(ticker, rmse, max_clip, hidden_dim, activation_function))

                tr_rmse = rmse

                # predict
                preds = []
                for i in range(Y_val.shape[0]):
                    x, y = X_val[i], Y_val[i]
                    prev_s = np.zeros((hidden_dim, 1))
                    # Forward pass
                    for t in range(T):
                        mulu = np.zeros((hidden_dim, 1))
                        # # print(x.shape[1])
                        for k in range(x.shape[1]):
                            new_dim = np.expand_dims(x[:, k], axis=1)
                            mulu += np.dot(U[:, :], new_dim)
                        # print(U[:,:,k].shape)
                        mulw = np.dot(W, prev_s)
                        add = mulw + mulu + bh
                        if activation_function == 'sigmoid':
                            s = sigmoid_forward(add)
                        if activation_function == 'tanh':
                            s = tanh_forward(add)
                        if activation_function == 'relu':
                            s = relu_forward(add)
                        mulv = np.dot(V, s) + by
                        prev_s = s
                    preds.append(mulv)

                preds = np.array(preds)

                preds = np.reshape(preds, (len(preds), 1))
                preds_like = np.zeros(shape=(len(preds), 6))
                preds_like[:, 0] = preds[:, 0]
                preds = sc.inverse_transform(preds_like)[:, 0]

                train = data[T:training_data_len]
                valid = data[training_data_len:]
                valid['Predictions'] = preds
                #print(valid)
                rmse = np.sqrt(mean_squared_error(valid['Predictions'], valid['Close']))
                print('testing rmse = ', rmse)


                #axis[0,1].set_figure(figsize=(12, 6))
                axis[0,1].set_title('Testing Ticker model: {}, rmse= {}, clip max= {}, h_dim= {}, {}'.format(ticker, rmse, max_clip, hidden_dim, activation_function))
                axis[0,1].set_xlabel('Date')
                axis[0,1].set_ylabel('Close Price')
                # plt.plot(train['Close'])
                axis[0,1].plot(valid[['Close', 'Predictions']])
                axis[0,1].legend(['Testing data', 'Predictions'], loc='lower left')
                #plt.savefig('Testing {} rmse= {} Cmax= {} Hdim= {} {}.jpg'.format(ticker, rmse, max_clip, hidden_dim, activation_function))

                #axis[1, 0].set_figure(figsize=(12, 6))
                axis[1, 0].set_title('Loss Ticker model: {} clip max = {}'.format(ticker, max_clip))
                axis[1, 0].set_xlabel('epoch')
                axis[1, 0].set_ylabel('Loss')
                # plt.plot(train['Close'])
                axis[1, 0].plot(loss_arr)
                axis[1, 0].plot(val_loss_arr)
                axis[1, 0].legend(['train_loss', 'validation_loss'], loc='upper right')
                figure1.savefig('{} Cmax= {} Hdim= {} Acti= {}.jpg'.format(ticker, max_clip, hidden_dim, activation_function))

                table[c][0] = str(c+1)
                table[c][1] = 'Number of neurons = ' + str(hidden_dim) + '\nError Function = RMSE\n' + 'Activation function = ' + str(activation_function) +'\nMin and Max Clipping = ' + str(min_clip) + ', ' + str(max_clip)
                table[c][2] = 'Train/Test Split = 80:20 \n Size of dataset = 504 \n Training RMSE = ' + str(tr_rmse) +'\n Test RMSE = '+ str(rmse)
                c+=1

        fig1, ax1 = plt.subplots(figsize=(14, 20))
        the_table = ax1.table(cellText=table,
                              colLabels=['Experiment Number', 'Parameters Chosen', 'Results'],
                              loc='center')
        the_table.scale(1,7)
        fig1.savefig(str(activation_function) + 'Table for ' + str(ticker) + '.jpg')