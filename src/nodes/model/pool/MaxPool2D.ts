import { Node } from '@baklavajs/core';
import { Layers, Nodes } from '@/nodes/model/Types';

export default class MaxPool2D extends Node {
  type = Layers.Pool;
  name = Nodes.MaxPool2D;

  constructor() {
    super();
    this.addInputInterface('Input');
    this.addOutputInterface('Output');

    // TODO: Keras+Pytorch allow shortcut for specifying single int for all dimensions
    this.addOption('Kernel Size Height', 'IntegerOption');
    this.addOption('Kernel Size Width', 'IntegerOption');
    this.addOption('Stride Height', 'IntegerOption');
    this.addOption('Stride Width', 'IntegerOption');

    this.addOption('Padding', 'DropdownOption', 'Valid', undefined, {
      items: ['Valid', 'Same'],
    });
  }
}
