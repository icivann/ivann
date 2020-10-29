export const enum Layers {
  Linear = 'Linear',
  Conv = 'Convolution',
  Pool = 'Pooling',
  Regularization = 'Regularization',
  Reshape = 'Reshaping',
  Custom = 'Custom',
  IO = 'I/O',
}

export const enum Nodes {
  Dense = 'Dense',
  Conv1D = 'Convolution1D',
  Conv2D = 'Convolution2D',
  Conv3D = 'Convolution3D',
  MaxPool2D = 'MaxPooling2D',
  Dropout = 'Dropout',
  Flatten = 'Flatten',
  Custom = 'Custom',
  InModel = 'InModel',
  OutModel = 'OutModel',
}
