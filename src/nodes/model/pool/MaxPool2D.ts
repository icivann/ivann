import { Node } from '@baklavajs/core';
import { Layers, Nodes } from '@/nodes/model/Types';
import { valuesOf } from '@/app/util';
import { Padding } from '@/app/ir/irCommon';

export default class MaxPool2D extends Node {
  type = Layers.Pool;
  name = Nodes.MaxPool2D;

  constructor() {
    super();
    this.addInputInterface('Input');
    this.addOutputInterface('Output');

    // TODO: Keras+Pytorch allow shortcut for specifying single int for all dimensions
    this.addOption('Kernel Size', 'VectorOption', [1, 1]);
    this.addOption('Stride', 'VectorOption', [1, 1]);

    this.addOption('Padding', 'DropdownOption', 'Valid', undefined, {
      items: valuesOf(Padding),
    });
  }
}
