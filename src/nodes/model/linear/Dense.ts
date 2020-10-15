import { Node } from '@baklavajs/core';
import { Layers, Nodes } from '@/nodes/model/Types';

export default class Dense extends Node {
  type = Layers.Linear;
  name = Nodes.Dense;

  constructor() {
    super();
    this.addInputInterface('Input');
    this.addOutputInterface('Output');

    this.addOption('Size', 'IntegerOption');
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
