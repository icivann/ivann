import { MlNode } from '@/app/ir/mainNodes';
import Conv2D from '@/app/ir/conv/Conv2D';

export const enum Layers {
  Linear = 'Linear',
  Conv = 'Convolution',
  Pool = 'Pooling',
  Regularization = 'Regularization',
  Reshape = 'Reshaping',
}

export const enum Nodes {
  Dense = 'Dense',
  Conv2D = 'Convolution2D',
  MaxPool2D = 'MaxPooling2D',
  Dropout = 'Dropout',
  Flatten = 'Flatten'
}

type Options = Map<string, any>
export const nodeBuilder: Map<String, (r:Options) => MlNode> = new Map([
  ["Convolution2D", Conv2D.build]
]);
