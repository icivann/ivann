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
    this.addOption('Use Bias', 'CheckboxOption', 'True');
    this.addOption('Weights initializer', 'InputOption'); // TODO check
    this.addOption('Bias regularizer', 'InputOption'); // TODO check
    this.addOption('Weights regularizer', 'InputOption'); // TODO check
  }
}
