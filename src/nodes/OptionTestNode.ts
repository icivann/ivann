import { Node } from '@baklavajs/core';

export default class OptionTestNode extends Node {
  type = 'Layer';
  name = 'OptionTestNode';

  constructor() {
    super();
    this.addInputInterface('Input');
    this.addOutputInterface('Output');
    this.addOption('Vector x', 'IntegerOption', 32);
    this.addOption('Vector y', 'IntegerOption', 3);
    this.addOption('Checkbox', 'CheckboxOption', true);

    this.addOption('VectorOption', 'VectorOption', [3, 4, 7]);
    this.addOption('IntegerOption', 'Integer', 32);
  }

  public calculate() {
    this.getInterface('Output').value = 0;
  }
}
