import { Node } from '@baklavajs/core';
import { Nodes } from '@/nodes/train/Types';

export enum DatasetOptions {
  Dataset = 'Dataset',
}

export default class DatasetInput extends Node {
  type = Nodes.DatasetInput;
  name = Nodes.DatasetInput;

  constructor() {
    super();
    this.addOption(DatasetOptions.Dataset, 'DropdownOption', 'Valid', undefined, {
      items: ['MNIST'],
    });
    this.addOutputInterface('x');
    this.addOutputInterface('y');
  }
}
