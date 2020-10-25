import { Node } from '@baklavajs/core';
import { Layers, Nodes } from '@/nodes/model/Types';
import { valuesOf } from '@/app/util';
import { Padding } from '@/app/ir/irCommon';
import { Conv2DOptions } from '@/nodes/model/conv/Conv2D';

export enum MaxPool2DOptions{
  KernelSize = 'Kernel Size',
  Stride = 'Stride',
  Padding = 'Padding'
}
export default class MaxPool2D extends Node {
  type = Layers.Pool;
  name = Nodes.MaxPool2D;

  constructor() {
    super();
    this.addInputInterface('Input');
    this.addOutputInterface('Output');

    // TODO: Keras+Pytorch allow shortcut for specifying single int for all dimensions
    this.addOption(MaxPool2DOptions.KernelSize, 'VectorOption', [1, 1], undefined, {
      min: [1, 1],
    });
    this.addOption(MaxPool2DOptions.Stride, 'VectorOption', [1, 1],  undefined, {
      min: [1, 1],
    });
    this.addOption(Conv2DOptions.Padding, 'DropdownOption', 'Valid', undefined, {
      items: valuesOf(Padding),
    });
  }
}
