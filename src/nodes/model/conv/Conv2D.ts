import { Node } from '@baklavajs/core';

export default class Conv2D extends Node {
  type = 'Layer';

  name = 'Conv2D';

  constructor() {
    super();
    this.addInputInterface('Input');
    this.addOutputInterface('Output');
    this.addOption('Filters', 'IntegerOption', 32);
    this.addOption('Kernel Size x', 'IntegerOption', 3);
    this.addOption('Kernel Size y', 'IntegerOption', 3);
    this.addOption('Activation', 'SelectOption', 'ReLU', undefined, {
      selected: 'ReLU',
      items: ['ReLU', 'Tanh', 'Sigmoid', 'Linear'],
    });
    this.addOption('Input Shape x', 'IntegerOption', 28);
    this.addOption('Input Shape y', 'IntegerOption', 28);
    this.addOption('Input Shape z', 'IntegerOption', 3);
  }

  public calculate() {
    this.getInterface('Output').value = 1423;
  }
}
