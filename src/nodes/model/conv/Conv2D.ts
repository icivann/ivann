import { Node } from '@baklavajs/core';
import { Layers, Nodes } from '@/nodes/model/Types';

export default class Conv2D extends Node {
  type = Layers.Conv;
  name = Nodes.Conv2D;

  constructor() {
    super();
    this.addInputInterface('Input');
    this.addOutputInterface('Output');

    this.addOption('Filters', 'IntegerOption');

    // TODO: Keras+Pytorch allow shortcut for specifying single int for all dimensions
    this.addOption('Kernel Size Height', 'IntegerOption');
    this.addOption('Kernel Size Width', 'IntegerOption');
    this.addOption('Stride Height', 'IntegerOption');
    this.addOption('Stride Width', 'IntegerOption');

    this.addOption('Padding', 'DropdownOption', 'Valid', undefined, {
      items: ['Valid', 'Same'],
    });
    this.addOption('Activation', 'DropdownOption', 'None', undefined, {
      items: ['None', 'ReLU', 'Tanh', 'Sigmoid', 'Linear'],
    });
    this.addOption('Use Bias', 'CheckboxOption', true);

    // TODO: Decide default value and options for these
    this.addOption('Weights Initializer', 'DropdownOption', 'Xavier', undefined, {
      items: ['Xavier'],
    });
    this.addOption('Bias Initializer', 'DropdownOption', 'Zeros', undefined, {
      items: ['Zeros', 'Ones'],
    });
    this.addOption('Bias Regularizer', 'DropdownOption', 'None', undefined, {
      items: ['None'],
    });
    this.addOption('Weights Regularizer', 'DropdownOption', 'None', undefined, {
      items: ['None'],
    });
  }
}
