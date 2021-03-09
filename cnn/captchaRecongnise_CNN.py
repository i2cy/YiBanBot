#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author: i2cy(i2cy@outlook.com)
# Filename: captchaRecongnise_CNN.py
# Created on: 2020/10/17

import os, time, psutil
import matplotlib.pyplot as plt
import random
import pathlib
import json

# 不使用GPU
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# *屏蔽tensorflow警告信息输出
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

# *RTX硬件兼容性修改配置
if len(tf.config.list_physical_devices('GPU')) > 0:
    tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)


DATASET_ROOT = "../../DeepLearning/Datasets/captcha_CN/"    # 在此处修改数据集文件夹
TEST_RATE = 0.05                                            # 测试集比例
BATCH_SIZE = 10                                             # 批处理大小
EPOCHES = 10                                                # 训练代数
BUFF_RATE = 0.1                                             # 缓冲区大小指数
LEARNING_RATE = 0.0001                                      # 学习率
MODEL_FILE = "Models/captchaRecongnise_model.h5"            # 在此处修改神经网络模型文件
NAME = "CaptchaRecongnise"
DICT_FILE = "words/dict/dict.json"                          # 字符集与标签的转换集文件（由神经网络接口部分自动生成）

CAPTCHA_CNN = None

LABEL_TO_WORD = {}


class customNN:
    def __init__(self, model_name="MLP"):
        self.name = model_name
        self.train_db = None
        self.test_db = None
        self.model = None
        self.train_size = 0
        self.test_size = 0
        self.data_shape = []
        self.batch_size = 8
        self.train_history = None
        self.tensorboard_enable = False
        self.log_root = "./tensorflow_log"
        self.callbacks = []
        self.callback_file_writer = None
        self.base_model = None
        self.epoch = 0
        self.model_file = "{}.h5".format(self.name)
        self.autosave = False
        self.output_counts = 0
        self.finetune_level = 0

        self.call_times = 0


    def _get_freeRAM(self):
        free_ram = psutil.virtual_memory().free
        return free_ram


    def _init_tensorboard(self):
        log_dir = os.path.join(self.log_root,
                               time.strftime("%Y%m%d-%H%M%S_")+
                               self.name
                               )
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir,
                                                              histogram_freq=1)
        self.callbacks.append(tensorboard_callback)
        self.callback_file_writer = tf.summary.create_file_writer(os.path.join(
                                                                  log_dir, "train"))
        self.callback_file_writer.set_as_default()


    def load_dataset(self, trainset, testset=None,
                     mapFunc=None, testRate=0.15, batchSize=8,
                     shufflePercentage=0.3, mapFuncTest=None,
                     mapFuncLabel=None, mapFuncLabelTest=None):  # dataset has to be formated tensors: (data, labels)
        self.batch_size = batchSize
        if testset == None:
            # randomly split trainset and testset
            datasets = [ele for ele in trainset]
            train_size = len(datasets[0]) - int(len(datasets[0]) * testRate)
            all_indexs = list(range(len(datasets[0])))
            random.shuffle(all_indexs)
            features = []
            labels = []
            if (type(datasets[1][0]) in (type([0]), type((0,)))) and len(datasets[1][0]) == len(all_indexs):
                for i in enumerate(datasets[1]):
                    labels.append([])
                    self.output_counts += 1
                for index in all_indexs[:train_size]:
                    data = datasets[0][index]
                    features.append(data)
                    for i, l in enumerate(datasets[1]):
                        label = datasets[1][i][index]
                        labels[i].append(label)
                if type(labels[0]) == type([0]):
                    labels = tuple(labels)
            else:
                self.output_counts += 1
                for index in all_indexs[:train_size]:
                    features.append(datasets[0][index])
                    labels.append(datasets[1][index])
            trainset = (features, labels)
            features = []
            labels = []
            if (type(datasets[1][0]) in (type([0]), type((0,)))) and len(datasets[1][0]) == len(all_indexs):
                for i in enumerate(datasets[1]):
                    labels.append([])
                for index in all_indexs[train_size:]:
                    data = datasets[0][index]
                    features.append(data)
                    for i, l in enumerate(datasets[1]):
                        label = datasets[1][i][index]
                        labels[i].append(label)
                if type(labels[0]) == type([0]):
                    labels = tuple(labels)
            else:
                for index in all_indexs[train_size:]:
                    features.append(datasets[0][index])
                    labels.append(datasets[1][index])
            testset = (features, labels)

        self.data_shape = tf.constant(trainset[0][0]).shape
        self.train_size = len(trainset[0])
        self.test_size = len(testset[0])

        print("trainset sample number: {}".format(str(self.train_size)))
        print("testset sample number: {}".format(str(self.test_size)))

        if mapFunc == None:
            if mapFuncLabel == None:
                train_db = tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(trainset[0]),
                                                tf.data.Dataset.from_tensor_slices(trainset[1])))
                test_db = tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(testset[0]),
                                               tf.data.Dataset.from_tensor_slices(testset[1])))
            else:
                if mapFuncLabelTest == None:
                    mapFuncLabelTest = mapFuncLabel
                train_db = tf.data.Dataset.zip((
                    tf.data.Dataset.from_tensor_slices(trainset[0]), tf.data.Dataset.from_tensor_slices(
                        trainset[1]).map(mapFuncLabel, num_parallel_calls=tf.data.experimental.AUTOTUNE)))

                test_db = tf.data.Dataset.zip((
                    tf.data.Dataset.from_tensor_slices(testset[0]), tf.data.Dataset.from_tensor_slices(
                        testset[1]).map(mapFuncLabelTest, num_parallel_calls=tf.data.experimental.AUTOTUNE)))

        else:
            if mapFuncTest == None:
                mapFuncTest = mapFunc
            self.data_shape = mapFunc(trainset[0][0]).shape
            train_db = tf.data.Dataset.from_tensor_slices(trainset[0])
            train_db = train_db.map(mapFunc, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            test_db = tf.data.Dataset.from_tensor_slices(testset[0])
            test_db = test_db.map(mapFuncTest)

            if mapFuncLabel == None:
                train_db = tf.data.Dataset.zip((
                    train_db, tf.data.Dataset.from_tensor_slices(trainset[1])))
                test_db = tf.data.Dataset.zip((
                    test_db, tf.data.Dataset.from_tensor_slices(testset[1])))
            else:
                if mapFuncLabelTest == None:
                    mapFuncLabelTest = mapFuncLabel
                train_db = tf.data.Dataset.zip((
                    train_db, tf.data.Dataset.from_tensor_slices(
                        trainset[1]).map(mapFuncLabel, num_parallel_calls=tf.data.experimental.AUTOTUNE)))

                test_db = tf.data.Dataset.zip((
                    train_db, tf.data.Dataset.from_tensor_slices(
                        testset[1]).map(mapFuncLabelTest, num_parallel_calls=tf.data.experimental.AUTOTUNE)))

        datasize = 1
        for size in self.data_shape:
            datasize *= size
        freeRAM = int(self._get_freeRAM() * shufflePercentage)
        print("free RAM size: {} MB".format(str(freeRAM//1048576)))

        shuffle_MaxbuffSize = int((freeRAM * 0.8)//datasize)
        prefetch_buffSize = int((freeRAM * 0.2)//(datasize * self.batch_size))

        print("automatically allocated data buffer size: {} MB".format(str(shuffle_MaxbuffSize*datasize//1048576)))

        shuffle_buffSize = shuffle_MaxbuffSize
        if shuffle_MaxbuffSize > self.train_size:
            shuffle_buffSize = self.train_size
        train_db = train_db.shuffle(shuffle_buffSize).repeat().batch(self.batch_size).prefetch(prefetch_buffSize)
        shuffle_buffSize = shuffle_MaxbuffSize
        if shuffle_MaxbuffSize > self.test_size:
            shuffle_buffSize = self.test_size
        test_db = test_db.shuffle(shuffle_buffSize).repeat().batch(self.batch_size).prefetch(prefetch_buffSize)

        self.train_db = train_db
        self.test_db = test_db


    def set_model_file(self, path):
        self.model_file = path


    def enable_tensorboard(self, log_dir_root="./tensorflow_log"):
        self.log_root = log_dir_root
        self.tensorboard_enable = True


    def enable_checkpointAutosave(self, path=None):
        if path != None:
            self.model_file = path
        checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=self.model_file)
        self.add_callback(checkpoint)
        self.autosave = True


    def add_callback(self, callback_func):  # all callbacks added will be reset after training
        self.callbacks.append(callback_func)


    def init_model(self):  # 神经网络模型

        self.base_model = tf.keras.applications.Xception(input_shape=self.data_shape,
                                                            include_top=False,
                                                            weights="imagenet")
        self.base_model.trainable = False

        model = tf.keras.Sequential([
            self.base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(4096, activation="relu"),
            tf.keras.layers.Dense(3500, activation="softmax")
        ])

        self.model = model

        self.compile_model()




    def postProc_model(self, finetune_level=None):  # 模型后期处理（微调）
        if finetune_level is None:
            finetune_level = self.finetune_level
        model = self.model
        learning_rate = LEARNING_RATE
        if finetune_level == 1:
            fine_tune_at = -27
            learning_rate /= 2
            self.finetune_level = 1
        elif finetune_level == 2:
            learning_rate /= 4
            fine_tune_at = -47
            self.finetune_level = 2
        else:
            fine_tine_at = -47


        model.layers[0].trainable = True

        for layer in model.layers[0].layers[:fine_tune_at]:
            layer.trainable = False

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      loss="sparse_categorical_crossentropy",  # 多分类问题
                      metrics=["acc"]
                      )

        self.model = model
        print(model.summary())


    def compile_model(self):
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                           loss="sparse_categorical_crossentropy",  # 多分类问题
                           metrics=["acc"]
                           )


    def save_model(self, path=None):
        if path != None:
            self.model_file = path
        self.model.save(self.model_file)


    def load_model(self, path=None):
        if path != None:
            self.model_file = path
        self.model = tf.keras.models.load_model(self.model_file, compile=True)
        self.compile_model()



    def train(self, epochs=100):
        if self.tensorboard_enable and self.epoch == 0:
            self._init_tensorboard()
        try:
            self.train_history = self.model.fit(self.train_db,
                                                epochs=epochs,
                                                initial_epoch=self.epoch,
                                                steps_per_epoch=self.train_size//self.batch_size,
                                                validation_data=self.test_db,
                                                validation_steps=self.test_size//self.batch_size,
                                                callbacks=self.callbacks
                                                )
            self.epoch += epochs
        except KeyboardInterrupt:
            print("\ntraining process stopped manually")
            if self.autosave:
                self.load_model(self.model_file)


    def show_history_curves(self):
        plt.plot(self.train_history.epoch, self.train_history.history["loss"], label="Loss_Train")
        plt.plot(self.train_history.epoch, self.train_history.history["acc"], label="Acc_Train")
        plt.plot(self.train_history.epoch, self.train_history.history["val_loss"], label="Loss_Test")
        plt.plot(self.train_history.epoch, self.train_history.history["val_acc"], label="Acc_Test")
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.title(self.name)
        plt.legend()
        plt.show()


    def _generate_bar(self, percent):
        res = " {}%\t[".format(str(round(percent*100, 1)))
        if percent == 1:
            res += "="*30
        else:
            done = int(30*percent)
            res += done*"="
            res += ">"
            res += (29 - done)*" "

        res += "]"
        return res


    def evaluate(self):
        print("evaluating model with test datasets...")

        acc = self.model.evaluate(self.test_db, return_dict=True,
                                  steps=self.test_size//self.batch_size)

        return acc




    def predict(self, data):
        try:
            if len(data.shape) != len(self.data_shape) + 1:
                data = tf.expand_dims(data, 0)
        except:
            pass
        res = self.model.predict(data)
        self.call_times += 1
        return res


def read_preprocess_image(img_path):  # 定义数据集map函数
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize_with_pad(img, 300, 300)
    img = tf.image.resize(img, [224, 224])
    img = tf.cast(img, tf.float32)
    img = img / 255  # 图像归一化，使得输入数据在（-1,1）区间范围内
    return img


def read_preprocess_image_from_raw(img):
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize_with_pad(img, 300, 300)
    img = tf.image.resize(img, [224, 224])
    img = tf.cast(img, tf.float32)
    img = img / 255  # 图像归一化，使得输入数据在（-1,1）区间范围内
    return img


class predictor:
    def __init__(self, dnn):
        self.dnn = dnn
        self.labels_converter = []
        self.pre_process_func = self._default_preprocess


    def _default_preprocess(self, data):
        return data


    def load_labels(self, label_names):  # label_names 必须为列表 [[标签列表1], [标签列表2]]
        for label in label_names:
            converter = dict((index, name)
                              for index, name in enumerate(label))
            self.labels_converter.append(converter)


    def set_preprocess_func(self, func):
        self.pre_process_func = func


    def predict(self, data):
        res_raw = self.dnn.predict(
            self.pre_process_func(data)
        )
        res = []
        for index, converter in enumerate(self.labels_converter):
            res.append(converter.get(tf.argmax(res_raw[index][0]).numpy()))
        return res


def path_fixer(path): # path checker
    chk = ""
    for i in path:
        chk += i
        if i in ("/", "\\"):
            if not os.path.exists(chk):
                os.mkdir(chk)


def captcha_recongnise(file_raw):
    global CAPTCHA_CNN, LABEL_TO_WORD
    res = CAPTCHA_CNN.predict(read_preprocess_image_from_raw(file_raw))
    word = LABEL_TO_WORD[str(tf.argmax(res[0]).numpy())]
    #print("CNN captcha result: {}".format(word))
    return word


def cnn_init():
    global CAPTCHA_CNN, LABEL_TO_WORD

    paths = [MODEL_FILE, DICT_FILE]
    for i in paths:
        path_fixer(i)

    print("initializing...")

    try:
        LABEL_TO_WORD = json.load(open(DICT_FILE, "r"))
    except Exception:
        print("warning: label to word diction not ready")

    if __name__ != "__main__":
        CAPTCHA_CNN = customNN(NAME)
        CAPTCHA_CNN.load_model(MODEL_FILE)
        print("captcha CNN fully connected")


def main():
    data_root = pathlib.Path(DATASET_ROOT)

    img_paths = [str(ele) for ele in data_root.glob("*.jpg")]

    labels = []
    label_to_word = {}
    for i in img_paths:
        i = i.split('/')[-1]
        name = i.split("_")
        id = int(name[1])
        label_to_word.update({id:name[0]})
        labels.append(id)

    print("dumping label_to_word...")
    json.dump(label_to_word, open(DICT_FILE,'w'), indent=2)

    img_counts = len(img_paths)
    print("loaded", img_counts, "image paths")

    label_counts = len(labels)
    print("loaded", label_counts, "labels")

    print("dataset head:")
    print("============================================" * 3)
    print("Label\t\tIMG_Path")
    for i, l in enumerate(labels):
        if i > 9:
            break
        print(" " + str(labels[i]) + "\t\t" + str(img_paths[i]))
    print("============================================" * 3)


    # 初始化神经网络
    cnn = customNN(NAME)
    cnn.load_dataset((img_paths, labels),
                     mapFunc=read_preprocess_image,
                     testRate=TEST_RATE,
                     batchSize=BATCH_SIZE,
                     shufflePercentage=BUFF_RATE
                     )

    # 初始化网络模型并执行设置
    if os.path.exists(MODEL_FILE):
        cnn.load_model(MODEL_FILE)
        print("loaded model file from \"{}\"".format(MODEL_FILE))
    else:
        cnn.init_model()

    print(cnn.model.summary())

    cnn.set_model_file(MODEL_FILE)


    print("testing model speed...")

    val_root = pathlib.Path("../validations")

    test_files = [str(ele) for ele in val_root.glob("*.jpg")]

    test_data = [read_preprocess_image(ele) for ele in test_files]

    for i in range(5):
        t1 = time.time()
        for ele in test_data:
            res = cnn.predict(ele)
        t2 = time.time()
        print("  testing {} files, time spent {}ms, {}ms per pred".format(len(test_data),
                                                  round((t2-t1)*1000, 2),
                                                  round(((t2-t1)/len(test_data))*1000, 2)))

    print("outputs: {}".format(res))

    for i in range(5):
        t3 = time.time()
        data = tf.data.Dataset.from_tensor_slices(test_data).repeat().batch(8)
        t4 = time.time()
        t1 = time.time()
        res = cnn.predict(data.take(1))
        t2 = time.time()
        print("  testing {} files, time spent {}ms, {}ms per data,\n  batching spent {}ms, total {}ms".format(len(test_data),
                                                                                              round((t2-t1)*1000, 2),
                                                                                              round((t2-t1)*125, 2),
                                                                                              round((t4-t3)*1000, 2),
                                                                                                  round((t2-t3)*1000, 2)))

    print("outputs: {}".format(res))


    cnn.enable_tensorboard()
    #cnn.enable_checkpointAutosave(MODEL_FILE)

    # 检查数据集匹配是否有错
    print("datasets:\n{}".format(str(cnn.train_db)))
    for i in range(3):
        img = read_preprocess_image(img_paths[random.randint(0, img_counts)])
        plt.imshow(img)
        plt.show()


    # 微调模型
    if True:
        choice = input("should we fine tune now? level(1,2/n): ".format(str(EPOCHES)))
        if EPOCHES > 0 and choice in ("1","2"):
            cnn.postProc_model(int(choice))

    # 初次训练网络
    choice = input("start training for {} epoch(s)? (Y/n): ".format(str(EPOCHES)))
    trained = False
    if EPOCHES > 0 and choice in ("Y", "y", "yes"):
        cnn.train(epochs=EPOCHES)
        trained = True

    # 保存模型
    if trained:
        cnn.save_model()
        print("model saved to \"{}\"".format(MODEL_FILE))

    # 测试模型
    print("evaluating trained model...")
    cnn.evaluate()


    # 预测
    while True:
        path = input("test file path: ")
        res = cnn.predict(read_preprocess_image(path))
        word = label_to_word[tf.argmax(res[0]).numpy()]
        print("result: {}".format(word))
        plt.imshow(read_preprocess_image(path))
        plt.show()


if __name__ == "__main__":
    cnn_init()
    main()
else:
    cnn_init()