import { Node } from '@baklavajs/core';
import { ModelNodes } from '@/nodes/model/Types';

export default class OutModel extends Node {
  type = ModelNodes.OutModel;
  name = ModelNodes.OutModel;

  constructor() {
    super();
    this.addInputInterface('Input');
  }
}
