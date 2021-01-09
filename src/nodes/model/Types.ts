export enum ModelCategories {
  SparseLayers = 'Sparse Layers',
  RecurrentLayers = 'Recurrent Layers',
  Normalization = 'Normalisation',
  DistanceFunctions = 'Distance Functions',
  Linear = 'Linear',
  Conv = 'Convolution',
  Pool = 'Pooling',
  Dropout = 'Dropout',
  Regularization = 'Regularization',
  Reshape = 'Reshaping',
  Activation = 'Activation Functions',
  Custom = 'Custom',
  Operations = 'Operations',
  IO = 'I/O',
  Transformer = 'Transformer',
  NonLinearActivation = 'Non-Linear Activation',
  Padding = 'Padding',
}

export enum ModelNodes {
  Fold = 'Fold',
  Unfold = 'Unfold',
  PairwiseDistance = 'PairwiseDistance',
  CosineSimilarity = 'CosineSimilarity',
  Embedding = 'Embedding',
  EmbeddingBag = 'EmbeddingBag',
  AlphaDropout = 'AlphaDropout',
  Conv1d = 'Conv1d',
  Conv2d = 'Conv2d',
  Conv3d = 'Conv3d',
  ConvTranspose1d = 'ConvTranspose1d',
  ConvTranspose2d = 'ConvTranspose2d',
  ConvTranspose3d = 'ConvTranspose3d',
  MaxPool1d = 'MaxPool1d',
  MaxPool2d = 'MaxPool2d',
  MaxPool3d = 'MaxPool3d',
  MaxUnpool1d = 'MaxUnpool1d',
  MaxUnpool2d = 'MaxUnpool2d',
  MaxUnpool3d = 'MaxUnpool3d',
  AvgPool1d = 'AvgPool1d',
  AvgPool2d = 'AvgPool2d',
  AvgPool3d = 'AvgPool3d',
  FractionalMaxPool2d = 'FractionalMaxPool2d',
  LPPool1d = 'LPPool1d',
  LPPool2d = 'LPPool2d',
  AdaptiveMaxPool1d = 'AdaptiveMaxPool1d',
  AdaptiveMaxPool2d = 'AdaptiveMaxPool2d',
  AdaptiveMaxPool3d = 'AdaptiveMaxPool3d',
  AdaptiveAvgPool1d = 'AdaptiveAvgPool1d',
  AdaptiveAvgPool2d = 'AdaptiveAvgPool2d',
  AdaptiveAvgPool3d = 'AdaptiveAvgPool3d',
  Dropout = 'Dropout',
  Dropout2d = 'Dropout2d',
  Dropout3d = 'Dropout3d',
  Relu = 'Relu',
  Softmax = 'Softmax',
  Flatten = 'Flatten',
  Concat = 'Concat',
  InModel = 'InModel',
  OutModel = 'OutModel',
  Bilinear = 'Bilinear',
  Linear = 'Linear',
  Softmin = 'Softmin',
  Transformer = 'Transformer',
  ReflectionPad1d = 'ReflectionPad1d',
  ReflectionPad2d = 'ReflectionPad2d',
  ReplicationPad1d = 'ReplicationPad1d',
  ReplicationPad2d = 'ReplicationPad2d',
  ReplicationPad3d = 'ReplicationPad3d',
  ZeroPad2d = 'ZeroPad2d',
  ConstantPad1d = 'ConstantPad1d',
  ConstantPad2d = 'ConstantPad2d',
  ConstantPad3d = 'ConstantPad3d',
  ELU = 'ELU',
  Hardshrink = 'Hardshrink',
  Hardsigmoid = 'Hardsigmoid',
  Hardtanh = 'Hardtanh',
  Hardswish = 'Hardswish',
  LeakyReLU = 'LeakyReLU',
  MultiheadAttention = 'MultiheadAttention',
  PReLU = 'PReLU',
  ReLU6 = 'ReLU6',
  RReLU = 'RReLU',
  SELU = 'SELU',
  CELU = 'CELU',
  GELU = 'GELU',
  Sigmoid = 'Sigmoid',
  SiLU = 'SiLU',
  Softplus = 'Softplus',
  Softshrink = 'Softshrink',
  Softsign = 'Softsign',
  Tanh = 'Tanh',
  LogSigmoid = 'LogSigmoid',
  Tanhshrink = 'Tanhshrink',
  Threshold = 'Threshold',

  LogSoftmax = 'LogSoftmax',
  Softmax2d = 'Softmax2d',
  AdaptiveLogSoftmaxWithLoss = 'AdaptiveLogSoftmaxWithLoss',

  BatchNorm1d = 'BatchNorm1d',
  BatchNorm2d = 'BatchNorm2d',
  BatchNorm3d = 'BatchNorm3d',
  GroupNorm = 'GroupNorm',
  SyncBatchNorm = 'SyncBatchNorm',
  InstanceNorm1d = 'InstanceNorm1d',
  InstanceNorm2d = 'InstanceNorm2d',
  InstanceNorm3d = 'InstanceNorm3d',
  LocalResponseNorm = 'LocalResponseNorm',

  TransformerEncoderLayer = 'TransformerEncoderLayer',
  TransformerDecoderLayer = 'TransformerDecoderLayer',

  RNNBase = 'RNNBase',
  RNN = 'RNN',
  LSTM = 'LSTM',
  GRU = 'GRU',
  RNNCell = 'RNNCell',
  LSTMCell = 'LSTMCell',
  GRUCell = 'GRUCell',

  ModelCustom = 'ModelCustom',
}
