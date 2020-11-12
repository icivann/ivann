import { Node } from '@baklavajs/core';
import { DataNodes } from '@/nodes/data/Types';

export default class ToTensor extends Node {
  type = DataNodes.ToTensor;
  name = DataNodes.ToTensor;

  constructor() {
    super();

    this.addInputInterface('Input');
    this.addOutputInterface('Output');
  }
}
