import { Node } from '@baklavajs/core';

export default class OptionTestNode extends Node {
  type = 'Layer';
  name = 'OptionTestNode';

  constructor() {
    super();
    this.addInputInterface('Input');
    this.addOutputInterface('Output');

    this.addOption('VectorOption', 'VectorOption', [3, 4, 7]);
    this.addOption('2ectorOption', 'IntegerOption', 4);
    this.addOption('3ectorOption', 'VectorOption', [3, 4, 7]);
  }

  public calculate() {
    this.getInterface('Output').value = 0;
  }
}
