import os

try:
    import numpy as np
except:
    os.system('pip install numpy==1.19.5')
    import numpy as np
try:
    from easydict import EasyDict as edict
except:
    os.system('pip install easydict==1.9')
    from easydict import EasyDict as edict
try:
    import tensorflow as tf
    from tensorflow.python.ops.confusion_matrix import confusion_matrix
except:
    os.system('pip install tensorflow==2.4.0')
    import tensorflow as tf
    from tensorflow.python.ops.confusion_matrix import confusion_matrix
try:
    import matplotlib.pylab as plt
except:
    os.system('pip install matplotlib==3.4.2')
    import matplotlib.pylab as plt
try:
    from umap import UMAP
except:
    os.system('pip install umap-learn==0.5.1')
    from umap import UMAP


class Model:
    """
    TrCNN模型主体
    """
    
    def __init__(self, config: dict, **kwargs):
        """

        Args:
            config: 配置文件
            **kwargs: 可更改关键字参数，关键字与config.keys()相同
        Returns:
            None
        """
        
        config.update(**kwargs)
        self.model_root = config.model_root
        self.weights_f_name = config.weights_f_name
        self.weights_c_name = config.weights_c_name
        self.input_length = config.input_length
        self.class_num = config.class_num
        self.method = config.method
        self.model_f = self.get_model_f()
        self.model_c = self.get_model_c()
        self.load_msg = self.load_weights()
        self.compile()
        self.batch_size = None        
    def get_model_f(self) -> tf.keras.Model:
        """
        Args:
            None
        Returns:
            特征提取模型

        """
        def res_bottleneck_block(x, filter1, filter2, strides, activation):
            b_layer_index = len(x)
            x.append(tf.keras.layers.Conv1D(
                filters=filter1, kernel_size=1, strides=1, padding='same', use_bias=False
            )(x[-1]))
            x.append(tf.keras.layers.BatchNormalization(axis=-1)(x[-1]))
            x.append(tf.keras.layers.Activation(activation=activation)(x[-1]))

            x.append(tf.keras.layers.Conv1D(
                filters=filter1, kernel_size=3, strides=strides, padding='same', use_bias=False
            )(x[-1]))
            x.append(tf.keras.layers.BatchNormalization(axis=-1)(x[-1]))
            x.append(tf.keras.layers.Activation(activation=activation)(x[-1]))

            x.append(tf.keras.layers.Conv1D(
                filters=filter2, kernel_size=1, strides=1, padding='same', use_bias=False
            )(x[-1]))
            x.append(tf.keras.layers.BatchNormalization(axis=-1)(x[-1]))

            if filter2 != x[b_layer_index].shape[-1] or strides != 1:
                x.append(tf.keras.layers.Conv1D(
                    filters=filter2, kernel_size=1, strides=strides, padding='same', use_bias=False
                )(x[b_layer_index]))
                x.append(tf.keras.layers.BatchNormalization(axis=-1)(x[-1]))
                x.append(tf.keras.layers.Add()([x[-3], x[-1]]))
            else:
                x.append(tf.keras.layers.Add()([x[b_layer_index], x[-1]]))
            x.append(tf.keras.layers.Activation(activation=activation)(x[-1]))
            return x

        x = [tf.keras.layers.Input(shape=(self.input_length, 1))]
        x.append(tf.keras.layers.Conv1D(
            filters=32, kernel_size=7, strides=1, padding='same', use_bias=False
        )(x[-1]))
        x.append(tf.keras.layers.Conv1D(
            filters=32, kernel_size=7, strides=1, padding='same', use_bias=False
        )(x[-1]))
        x.append(tf.keras.layers.Conv1D(
            filters=32, kernel_size=7, strides=1, padding='same', use_bias=False
        )(x[-1]))
        x.append(tf.keras.layers.MaxPool1D(pool_size=2, strides=2, padding='same')(x[-1]))
        x = res_bottleneck_block(x, filter1=32, filter2=128, strides=2, activation=tf.nn.leaky_relu)
        x = res_bottleneck_block(x, filter1=32, filter2=128, strides=1, activation=tf.nn.leaky_relu)

        x.append(tf.keras.layers.GlobalAvgPool1D()(x[-1]))
        x.append(tf.keras.layers.Dense(units=128)(x[-1]))
        model = tf.keras.models.Model(inputs=[x[0]], outputs=[x[-1]])
        model.build(input_shape=(self.input_length, 1))
        model.summary()
        return model

    def get_model_c(self) -> tf.keras.Model:
        """
        Args:
            None
        Returns:
            分类模型

        """
        x = [tf.keras.layers.Input(shape=(128,))]
        x.append(tf.keras.layers.Dense(units=64)(x[-1]))
        x.append(tf.keras.layers.Dense(units=self.class_num)(x[-1]))
        x.append(tf.keras.layers.Softmax()(x[-1]))
        model = tf.keras.models.Model(inputs=[x[0]], outputs=[x[-1]])
        model.build(input_shape=(128,))
        model.summary()
        return model 
    

    def compile(self):
        self.optimizer_f = tf.keras.optimizers.Adam(1e-3, beta_1=0.5, beta_2=0.999)
        self.optimizer_c = tf.keras.optimizers.Adam(1e-3, beta_1=0.5, beta_2=0.999)
        self.loss_fn_mmd = self.mmd
        self.loss_fn_pred = tf.keras.losses.SparseCategoricalCrossentropy()
        self.metrics = {
            **{
                k: tf.keras.metrics.Mean(k, dtype=tf.float32) for k in [
                    'loss_pred_s', 'loss_pred_t', 'loss_mmd'
                ]
            },
            **{
                k: tf.keras.metrics.SparseCategoricalAccuracy(k, dtype=tf.float32) for k in [
                    'acc_s', 'acc_t'
                ]
            }
        }

    @staticmethod
    def mmd(x, y, sigma, batch_size):
        """
        计算源域特征值和目标域特征值间的MMD距离

        Args:
            x: 源域数据特征
            y: 目标域数据特征
            sigma: 参数
            batch_size: 批大小

        Returns:

        """
        def RBF_function(x, y, sigma, batch_size):
            x_tensor = tf.expand_dims(x, -1)
            y_tensor = tf.expand_dims(tf.transpose(y), 0)
            diff_xy = tf.reduce_sum(tf.square(x_tensor - y_tensor), axis=1)
            a = tf.reduce_sum(tf.exp(-diff_xy / (2 * sigma))) / (batch_size ** 2)
            return a
        loss = (
            RBF_function(x, x, sigma, batch_size)
            + RBF_function(y, y, sigma, batch_size)
            - 2 * RBF_function(x, y, sigma, batch_size)
        )
        return loss

    def load_weights(self):
        """
        加载模型权重

        Args:
            None
        Returns:
            None

        """
        def load(weights_name, tar_model):
            model_file = os.path.join(self.model_root, weights_name)
            if os.path.exists(model_file):
                try:
                    tar_model.load_weights(filepath=model_file, by_name=True)
                    msg=f'加载模型权重文件成功'
                    print("--> 加载模型权重文件成功.")
                except Exception as e:
                    msg=f'加载模型权重文件失败'
                    print("--> 加载模型权重文件失败，以下为详细信息：")
                    print(e)
            else:
                msg=f'不存在{model_file}，跳过模型权重加载阶段'
                print(f"--> 不存在{model_file}，跳过模型权重加载阶段.")
            return msg
        msg_f=load(self.weights_f_name, self.model_f)
        msg_c=load(self.weights_c_name, self.model_c)
        return msg_f,msg_c
    @tf.function
    def train_step_with_mmd(self, xs, ys, xt, yt, sigma, batch_size):
        """
        单步训练函数，带有最大均值差异损失项，以适配源域与目标域特征分布，从而起到迁移源域诊断知识到目标域当中

        Args:
            xs: 源域数据
            ys: 源域标签
            xt: 目标域数据
            yt: 目标域标签（注意目标域标签未参与训练过程，只用于最终评估迁移模型在目标域的诊断精度）
            sigma: 高斯核参数
            batch_size: 训练批次大小

        Returns:

        """
        with tf.GradientTape(persistent=True) as tape:
            fs = self.model_f(xs, training=True)
            ft = self.model_f(xt, training=True)
            ys_pred = self.model_c(fs, training=True)
            yt_pred = self.model_c(ft, training=True)
            loss_mmd = self.loss_fn_mmd(fs, ft, sigma, batch_size)
            loss_pred_s = self.loss_fn_pred(ys, ys_pred)
            loss_pred_t = self.loss_fn_pred(yt, yt_pred)
            loss_all = 100 * loss_mmd + loss_pred_s
        grads_f = tape.gradient(loss_all, self.model_f.trainable_variables)
        grads_c = tape.gradient(loss_all, self.model_c.trainable_variables)
        del tape
        self.optimizer_f.apply_gradients(zip(grads_f, self.model_f.trainable_variables))
        self.optimizer_c.apply_gradients(zip(grads_c, self.model_c.trainable_variables))
        self.metrics['loss_pred_s'].update_state(loss_pred_s)
        self.metrics['loss_pred_t'].update_state(loss_pred_t)
        self.metrics['loss_mmd'].update_state(loss_mmd)
        self.metrics['acc_s'].update_state(ys, ys_pred)
        self.metrics['acc_t'].update_state(yt, self.model_c(self.model_f(xt, training=False), training=False))
        return {k: v.result() for k, v in self.metrics.items()}

    @tf.function
    def train_step_without_mmd(self, xs, ys, xt, yt, sigma, batch_size):
        """
        单步训练函数，无最大均值差异损失项，
        函数输入参数xt, yt, sigma与batch_size在函数中并未使用，目的在于与train_step_with_mmd函数的输出形式保持一致
        函数输出值中loss_mmd与loss_pred_t恒置为0，目的在于与train_step_with_mmd函数的输出形式保持一致

        Args:
            xs: 源域数据
            ys: 源域标签
            xt: 目标域数据
            yt: 目标域标签（注意目标域标签未参与训练过程，只用于最终评估迁移模型在目标域的诊断精度）
            sigma: 高斯核参数
            batch_size: 训练批次大小

        Returns:

        """
        with tf.GradientTape() as tape:
            fs = self.model_f(xs, training=True)
            ys_pred = self.model_c(fs, training=True)
            loss_pred_s = self.loss_fn_pred(ys, ys_pred)
        vars = self.model_f.trainable_variables + self.model_c.trainable_variables
        grads = tape.gradient(loss_pred_s, vars)
        self.optimizer_f.apply_gradients(zip(grads, vars))
        self.metrics['loss_pred_s'].update_state(loss_pred_s)
        self.metrics['acc_s'].update_state(ys, ys_pred)
        self.metrics['acc_t'].update_state(yt, self.model_c(self.model_f(xt, training=False), training=False))
        return {k: v.result() for k, v in self.metrics.items()}

    def reset_metrics(self):
        for k, v in self.metrics.items():
            v.reset_states()

    def train(
            self,
            xs: np.ndarray,
            ys: np.ndarray,
            xt: np.ndarray,
            yt: np.ndarray,
            epochs: int,
            batch_size: int,
            save_weights: bool
    ):
        """

        Args:
            x: 训练集数据
            y: 训练集数据对应标签
            epochs: 训练轮数
            batch_size: 训练中每轮对应批次大小
            save_weights: 是否保存模型

        Returns:
            res: 训练历史记录相关数据

        """
        self.batch_size = batch_size
        self.model_f.trainable = True
        self.model_c.trainable = True

        if self.method == "with_mmd":
            train_step = self.train_step_with_mmd
        elif self.method == "without_mmd":
            train_step = self.train_step_without_mmd
        else:
            print("训练类型输入不正确，method取值应为'with_mmd'或'without_mmd'")
            return

        # test_acc = 0.98
        # res = dict()
        data_s_iter = tf.data.Dataset.from_tensor_slices((xs, ys)).shuffle(xs.shape[0]).batch(
            self.batch_size, drop_remainder=True
        )
        data_t_iter = tf.data.Dataset.from_tensor_slices((xt, yt)).shuffle(xt.shape[0]).batch(
            self.batch_size, drop_remainder=True
        )
        for epoch in range(epochs):
            self.reset_metrics()
            for step, ((xs, ys), (xt, yt)) in enumerate(zip(data_s_iter, data_t_iter)):
                res = train_step(xs, ys, xt, yt, sigma=1, batch_size=self.batch_size)
                print(f"/rEpoch{epoch}-step{step}: " + ", ".join([f"{k}: {v:.4f}" for (k, v) in res.items()]), end='')
            print('')
            # ============= 当目标域数据集诊断准确度高于设定阈值时，更新保存的模型 =============
            # if epoch > 1 and test_acc < res['acc_t'] and save_weights:
            #     self.save_model_weights()
            #     test_acc = res['acc_t']
            # ==============================================================================
        if save_weights:
            self.save_model_weights()
        return None

    def save_model_weights(self):
        file_path = os.path.join(self.model_root, self.weights_f_name)
        self.model_f.save_weights(file_path)
        print(f"-->模型权重已成功保存至{file_path}.")
        file_path = os.path.join(self.model_root, self.weights_c_name)
        self.model_c.save_weights(file_path)
        print(f"-->模型权重已成功保存至{file_path}.")

    def get_conmat(self, y_true: np.ndarray, y_pred: np.ndarray, class_list: list) -> dict:
        """
        获取混淆矩阵

        Args:
            y_true: 真实标签向量
            y_pred: 预测标签向量
            class_list: 标签对应真实名称

        Returns:
            conmat_info: 混淆矩阵相关信息

        """
        conmat = tf.math.confusion_matrix(y_true, y_pred).numpy()
        precision, recall, f1score = [], [], []
        for i in range(self.class_num):
            precision.append(conmat[i, i] / (np.sum(conmat[:, i]) + 1e-5))
            recall.append(conmat[i, i] / (np.sum(conmat[i, :]) + 1e-5))
            f1score.append(2 * precision[i] * recall[i] / (precision[i] + recall[i] + 1e-5))
        accuracy = np.sum([conmat[i, i] for i in range(self.class_num)]) / np.sum(conmat)
        conmat_info = edict(
            class_list=class_list if class_list is not None else np.arange(self.class_num),
            conmat=conmat,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1score=f1score
        )
        return conmat_info

    def save_conmat(self, conmat: np.ndarray, class_list: list, save_path: str, domain: str):
        """
        保存混淆矩阵为图片

        Args:
            conmat: 混淆矩阵
            class_list: 标签列表
            save_path: 图片保存路径（包含文件夹及文件名）
            domain: 数据域描述
        """
        plt.figure(figsize=(5, 5))
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.imshow(conmat, cmap=plt.cm.Blues)
        plt.title(f"{'源域' if domain=='source' else '目标域'}混淆矩阵")
        indices = np.arange(conmat.shape[0])
        plt.xticks(indices, class_list, rotation=90)
        plt.yticks(indices, class_list)
        plt.colorbar(orientation='vertical', fraction=0.045, pad=0.05)
        plt.xlabel('预测值')
        plt.ylabel('真实值')
        for first_index in indices:
            for second_index in indices:
                number = conmat[second_index][first_index]
                color = 'white' if number > conmat.max() / 2. else 'black'
                plt.text(first_index-len(str(number))*0.04, second_index, number, color=color)
        save_path_pre_fix = save_path.split('.')[0]
        plt.savefig(f"{save_path_pre_fix}_conmat_{domain}.png", dpi=300, format='png', bbox_inches='tight')

    def get_scatter(self, xs, xt):
        features_xs = self.model_f.predict(xs)
        features_xt = self.model_f.predict(xt)
        features = np.concatenate([features_xs, features_xt], axis=0)
        features_umap = UMAP().fit_transform(features)
        return features_umap

    def save_scatter(self, xs, xt, ys, yt, class_list, save_path):
        data = self.get_scatter(xs, xt)
        plt.figure(figsize=(5, 5))
        colors = plt.get_cmap('tab10').colors
        data_xs, data_xt = np.split(data, axis=0, indices_or_sections=2)
        for i in range(len(class_list)):
            idx = np.where(ys==i)[0]
            plt.scatter(
                data_xs[idx, 0], data_xs[idx, 1], marker='o', alpha=0.5,
                s=35, edgecolor='black', linewidth=0.2, c=np.array(colors[i]).reshape(1, -1)
            )
        for i in range(len(class_list)):
            idx = np.where(yt==i)[0]
            plt.scatter(
                data_xt[idx, 0], data_xt[idx, 1], marker='+', alpha=1.,
                s=40, linewidth=0.2, c=np.array(colors[i]).reshape(1, -1)
            )
        class_list_s = [f"源域-{item}" for item in class_list]
        class_list_t = [f"目标域-{item}" for item in class_list]
        plt.legend(
            class_list_s + class_list_t,
            frameon=True, fontsize=12, ncol=2, bbox_to_anchor=(0, 1.2, 1.05, 0.2)
        )
        plt.xlabel("分量一", fontsize=12)
        plt.ylabel("分量二", fontsize=12)
        plt.title(f"分类散点分布图")
        plt.subplots_adjust(bottom=0.15, top=0.9, left=0.15, right=0.9, wspace=0., hspace=0)
        save_path_pre_fix = save_path.split('.')[0]
        plt.savefig(f"{save_path_pre_fix}_scatter.png", dpi=300, format='png', bbox_inches='tight')

    def evaluate(
        self, xs: np.ndarray, ys: np.ndarray, xt: np.ndarray, yt: np.ndarray,
        class_list: list, save_fig: bool, save_path: str
    ) -> tuple:
        """
        Args:
            xs：源域数据
            xt: 目标域数据
            ys: 源域数据对应标签
            yt: 目标域数据对应标签
            class_list: 类别名称列表
            save_fig: 是否保存图片
            save_path: 图片保存路径（包含保存文件名）

        Returns:
            res_source: 源域混淆矩阵相关信息
            res_target: 目标域混淆矩阵相关信息
        """
        def single_domain_evaluate(x, y, class_list, domain):
            pred_f = self.model_f.predict(x, batch_size=32)
            pred_c = self.model_c.predict(pred_f, batch_size=32).argmax(axis=1)
            if class_list is None:
                class_list = [str(i) for i in range(self.class_num)]
            res = self.get_conmat(y, pred_c, class_list)
            if save_fig:
                class_list, conmat, *_ = res.values()
                self.save_conmat(class_list=class_list, conmat=conmat, save_path=save_path, domain=domain)
            return res
        res_source = single_domain_evaluate(xs, ys, class_list, 'source')
        res_target = single_domain_evaluate(xt, yt, class_list, 'target')
        if save_fig:
            self.save_scatter(xs=xs, xt=xt, ys=ys, yt=yt, class_list=class_list, save_path=save_path)
        return res_source, res_target

    def predict(self, x: np.ndarray, class_list: list = None) -> dict:
        """

        Args:
            x: 待预测数据
            class_list: 类别名称列表

        Returns:
            prob: 各类别预测概率

        """
        if self.model_f.trainable:
            self.model_f.trainable = False
        if self.model_c.trainable:
            self.model_c.trainable = False
        x = np.squeeze(x)[None, :, None]
        pred_f = self.model_f.predict(x)
        pred_c = self.model_c.predict(pred_f).squeeze()
        if class_list is None:
            class_list = [str(i) for i in range(self.class_num)]
        prob = dict(zip(class_list, pred_c))
        return prob
