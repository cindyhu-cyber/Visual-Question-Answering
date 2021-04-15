# 基于VIS+LSTM的视觉问答模型


## 视觉问答模型介绍

思悦使用基于VIS+LSTM视觉问答模型为主播搭建智能协同平台，帮助主播实现对粉丝、用户问题的实时应答、自动应答、一键选择应答。
与常见的QA问答系统相比，用户不仅可以输入简单的纯文字或图像询问，而且可以同步输入图片+文字组合，模型可给出预测的答案。

## 模型用法

### 提取图像特征

```
th extract_fc7.lua -split train
th extract_fc7.lua -split val
```

#### 默认参数

- `batch_size`: Batch size. Default is 10.
- `split`: train/val. Default is `train`.
- `gpuid`: 0-indexed id of GPU to use. Default is -1 = CPU.
- `proto_file`: Path to the `deploy.prototxt` file for the VGG Caffe model. Default is `models/VGG_ILSVRC_19_layers_deploy.prototxt`.
- `model_file`: Path to the `.caffemodel` file for the VGG Caffe model. Default is `models/VGG_ILSVRC_19_layers.caffemodel`.
- `data_dir`: Data directory. Default is `data`.
- `feat_layer`: Layer to extract features from. Default is `fc7`.
- `input_image_dir`: Image directory. Default is `data`.


### 模型训练

```
th train.lua
```

#### 默认参数

- `rnn_size`: Size of LSTM internal state. Default is 512.
- `num_layers`: Number of layers in LSTM
- `embedding_size`: Size of word embeddings. Default is 512.
- `learning_rate`: Learning rate. Default is 4e-4.
- `learning_rate_decay`: Learning rate decay factor. Default is 0.95.
- `learning_rate_decay_after`: In number of epochs, when to start decaying the learning rate. Default is 15.
- `alpha`: Alpha for adam. Default is 0.8
- `beta`: Beta used for adam. Default is 0.999.
- `epsilon`: Denominator term for smoothing. Default is 1e-8.
- `batch_size`: Batch size. Default is 64.
- `max_epochs`: Number of full passes through the training data. Default is 15.
- `dropout`:  Dropout for regularization. Probability of dropping input. Default is 0.5.
- `init_from`: Initialize network parameters from checkpoint at this path.
- `save_every`: No. of iterations after which to checkpoint. Default is 1000.
- `train_fc7_file`: Path to fc7 features of training set. Default is `data/train_fc7.t7`.
- `fc7_image_id_file`: Path to fc7 image ids of training set. Default is `data/train_fc7_image_id.t7`.
- `val_fc7_file`: Path to fc7 features of validation set. Default is `data/val_fc7.t7`.
- `val_fc7_image_id_file`: Path to fc7 image ids of validation set. Default is `data/val_fc7_image_id.t7`.
- `data_dir`: Data directory. Default is `data`.
- `checkpoint_dir`: Checkpoint directory. Default is `checkpoints`.
- `savefile`: Filename to save checkpoint to. Default is `vqa`.
- `gpuid`: 0-indexed id of GPU to use. Default is -1 = CPU.

### 测试模型

```
th predict.lua -checkpoint_file checkpoints/vqa_epoch23.26_0.4610.t7 -input_image_path data/train2014/COCO_train2014_000000405541.jpg -question 'What is the cat on?'
```

#### 参数设置

- `checkpoint_file`: Path to model checkpoint to initialize network parameters from
- `input_image_path`: Path to input image
- `question`: Question string

## 模型预测结果

从VQA测试集中随机抽取图像+文字组合问题对，给出基于VIS+LSTM的视觉问答模型预测的答案。

![Image1](./data/image1.png)

Q: 站着的女生穿的衣服是什么颜色?
A: 红褐色

![Image2](./data/image2.png)

Q: 女主播手里拿着什么?
A: 蓝色牛仔裤

![Image3](./data/image3.png)

Q: 女生手里拿的是什么?
A: 土豆

## 参考

- [Exploring Models and Data for Image Question Answering], Ren et al., NIPS15
- [VQA: Visual Question Answering], Antol et al., ICCV15

