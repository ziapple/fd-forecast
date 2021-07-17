import keras
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.utils import to_categorical
from core import cwru_input_emd

# 迭代次数
EPOCHS = 50
# 每批次读取的训练数据集大小
BATCH_SIZE = 100


# 建立Sequential网络模型
def build_model(input_shape=(cwru_input_emd.IMF_X_LENGTH, cwru_input_emd.TIME_PERIODS), num_classes=cwru_input_emd.LABEL_SIZE):
    """
    LSTM(output_dim=CELL_SIZE, input_dim=INPUT_SIZE, input_length=TIME_STEPS, return_sequences=Flase，stateful=FALSE)
        output_dim：输出单个样本的特征值的维度
        input_dim： 输入单个样本特征值的维度
        input_length： 输入的时间点长度
        return_sequences：布尔值，默认False，控制返回类型。若为True则返回整个序列，否则仅返回输出序列的最后一个输出，即当return_sequences取值为True时，网络输入和输出的时间长度TIME_STEPS保持不变，而当return_sequences取值为FALSE时，网络输出的数据时间长度为1。例如输入数据时间长度为5，输出为一个结果。
        stateful： 布尔值，默认为False，若为True，则一个batch中下标为i的样本的最终状态将会用作下一个batch同样下标的样本的初始状态。当batch之间的时间是连续的时候，就需要stateful取True，这样batch之间时间连续。
    :param input_shape:
    :param num_classes:
    :return:
    """
    model = Sequential()
    model.add(LSTM(output_dim=256, input_shape=input_shape, activation='relu', return_sequences=True))
    model.add(LSTM(output_dim=256, return_sequences=True))
    model.add(LSTM(output_dim=256))
    model.add(Dense(256, activation='relu'))  # FC2 1024
    model.add(Dropout(rate=0.25))
    model.add(Dense(num_classes, activation='softmax'))  # Output 10
    print(model.summary())
    return model


def model_train():
    x_train, y_train, x_test, y_test = cwru_input_emd.read_emd_to_normal()
    print(x_train.shape)
    y_train = to_categorical(y_train, cwru_input_emd.LABEL_SIZE)
    y_test = to_categorical(y_test, cwru_input_emd.LABEL_SIZE)
    ckpt = keras.callbacks.ModelCheckpoint(filepath='../model/emd_model.{epoch:02d}-{val_loss:.4f}.h5',
                                           monitor='val_loss', save_best_only=True, verbose=1)
    model = build_model()
    opt = Adam(0.0002)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    print(model.summary())

    model.fit(x=x_train, y=y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.3, callbacks=[ckpt])
    model.save("finishModel_lstm.h5")


def model_valid():
    _, _, x_test, y_test = cwru_input_emd.read_emd_to_normal()
    y_test = to_categorical(y_test, cwru_input_emd.LABEL_SIZE)
    model = load_model('finishModel_lstm.h5')
    _y = model.predict(x_test)
    acc = np.equal(np.argmax(_y, axis=1), np.argmax(y_test, axis=1))
    print(np.sum(acc) / y_test.shape[0])


if __name__ == "__main__":
    # model_train()
    # model_valid()
    build_model()


