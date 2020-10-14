import { Node } from '@baklavajs/core';

export default class MaxPooling2D extends Node {
  type = 'Layer';

  name = 'MaxPooling2D';

  constructor() {
    super();
    this.addInputInterface('Input');
    this.addOutputInterface('Output');
    this.addOption('Pool Size x', 'IntegerOption', 2);
    this.addOption('Pool Size y', 'IntegerOption', 2);
  }

  public calculate() {
    this.getInterface('Output').value = 14235;
  }
}
