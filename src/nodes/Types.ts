export const enum Layers {
  Linear = 'Linear',
  Conv = 'Convolution',
  Pool = 'Pooling',
  Regularization = 'Regularization',
  Reshape = 'Reshaping',
  Custom = 'Custom',
  Operations = 'Operations',
  IO = 'I/O',
}

export const enum OverviewLayers {
  Train = 'Train',
  Optimizers = 'Optimizers',
}

export const enum OverviewNodes {
  Model = 'ModelNode',
  TrainClassifier = 'TrainClassifier',
  Adadelta = 'Adadelta',
}

export const enum Nodes {
  Dense = 'Dense',
  Relu = 'Relu',
  Softmax = 'Softmax',
  Conv1D = 'Convolution1D',
  Convtranspose1d = 'Convtranspose1d',
  Convtranspose2d = 'Convtranspose2d',
  Convtranspose3d = 'Convtranspose3d',
  Conv2D = 'Convolution2D',
  Conv3D = 'Convolution3D',
  Dropout2d = 'Dropout2d',
  Dropout3d = 'Dropout3d',
  MaxPool1D = 'MaxPooling1D',
  MaxPool2D = 'MaxPooling2D',
  MaxPool3D = 'MaxPooling3D',
  Dropout = 'Dropout',
  Flatten = 'Flatten',
  Custom = 'Custom',
  Concat = 'Concat',
  InModel = 'InModel',
  OutModel = 'OutModel',
  TrainClassifier = 'TrainClassifier',
}
