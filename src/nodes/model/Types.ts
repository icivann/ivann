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

export const enum Overview {
  ModelNode = 'ModelNode',
}

export const enum Nodes {
  Dense = 'Dense',
  Relu = 'Relu',
  Softmax = 'Softmax',
  Conv1D = 'Conv1D',
  Conv2D = 'Conv2D',
  Conv3D = 'Conv3D',
  ConvTranspose1d = 'ConvTranspose1d',
  ConvTranspose2d = 'ConvTranspose2d',
  ConvTranspose3d = 'ConvTranspose3d',
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
}
