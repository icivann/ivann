import { Node } from '@baklavajs/core';

export default class Dense extends Node {
    type = 'Dense';

    name = 'Dense';

    constructor() {
      super();
      this.addInputInterface('Number 1', 'NumberOption', 1);
      this.addInputInterface('Number 2', 'NumberOption', 10);
      this.addOutputInterface('Output');
      this.addOption('Operation', 'SelectOption', {
        selected: 'Add',
        items: ['Add', 'Subtract'],
      });
    }

    public calculate() {
      const n1 = this.getInterface('Number 1').value;
      const n2 = this.getInterface('Number 2').value;
      const operation = this.getOptionValue('Operation').selected;
      let result;
      if (operation === 'Add') {
        result = n1 + n2;
      } else if (operation === 'Subtract') {
        result = n1 - n2;
      }
      this.getInterface('Output').value = result;
    }
}
