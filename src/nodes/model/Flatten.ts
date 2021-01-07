import { Node } from '@baklavajs/core';
import { ModelNodes } from '@/nodes/model/Types';

export default class Flatten extends Node {
  type = ModelNodes.Flatten;
  name = ModelNodes.Flatten;

  constructor() {
    super();
    this.addInputInterface('Input');
    this.addOutputInterface('Output');
  }
}
