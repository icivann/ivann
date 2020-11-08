export const enum Layers {
  Linear = 'Linear',
  Conv = 'Convolution',
  Pool = 'Pooling',
  Dropout = 'Dropout',
  Regularization = 'Regularization',
  Reshape = 'Reshaping',
  Activation = 'Activation functions',
  Custom = 'Custom',
  Operations = 'Operations',
  IO = 'I/O',
}

export const enum Overview {
  ModelNode = 'ModelNode',
}

export const enum Nodes {
  Conv1D = 'Conv1D',
  Conv2D = 'Conv2D',
  Conv3D = 'Conv3D',
  ConvTranspose1d = 'ConvTranspose1d',
  ConvTranspose2d = 'ConvTranspose2d',
  ConvTranspose3d = 'ConvTranspose3d',
  MaxPool1D = 'MaxPool1d',
  MaxPool2D = 'MaxPool2d',
  MaxPool3D = 'MaxPool3d',
  Dropout = 'Dropout',
  Dropout2d = 'Dropout2d',
  Dropout3d = 'Dropout3d',
  Dense = 'Dense',
  Relu = 'Relu',
  Softmax = 'Softmax',
  Flatten = 'Flatten',
  Custom = 'Custom',
  Concat = 'Concat',
  InModel = 'InModel',
  OutModel = 'OutModel',
}
