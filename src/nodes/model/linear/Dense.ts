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
    this.addOption('Activation', 'SelectOption', 'None', undefined, {
      items: ['None', 'ReLU', 'Tanh', 'Sigmoid', 'Linear'],
    });
    this.addOption('Use Bias', 'CheckboxOption', true);

    // TODO: Decide default value and options for these
    this.addOption('Weights Initializer', 'SelectOption', 'Xavier', undefined, {
      items: ['Xavier'],
    });
    this.addOption('Bias Initializer', 'SelectOption', 'Zeros', undefined, {
      items: ['Zeros', 'Ones'],
    });
    this.addOption('Bias Regularizer', 'SelectOption', 'None', undefined, {
      items: ['None'],
    });
    this.addOption('Weights Regularizer', 'SelectOption', 'None', undefined, {
      items: ['None'],
    });
  }
}
