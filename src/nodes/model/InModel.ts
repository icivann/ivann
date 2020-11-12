import { Node } from '@baklavajs/core';
import { ModelNodes } from '@/nodes/model/Types';

export default class InModel extends Node {
  type = ModelNodes.InModel;
  name = ModelNodes.InModel;

  constructor() {
    super();

    this.addOutputInterface('Output');
  }
}
