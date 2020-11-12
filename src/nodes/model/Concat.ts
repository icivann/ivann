import { Node } from '@baklavajs/core';
import { ModelNodes } from '@/nodes/model/Types';

export default class Concat extends Node {
  type = ModelNodes.Concat;
  name = ModelNodes.Concat;

  constructor() {
    super();

    this.addInputInterface('Input 1');
    this.addInputInterface('Input 2');
    this.addOutputInterface('Output');
  }
}
