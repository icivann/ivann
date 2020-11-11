import { Node } from '@baklavajs/core';
import { Nodes } from '@/nodes/model/Types';

export default class InModel extends Node {
  type = Nodes.InModel;
  name = Nodes.InModel;

  constructor() {
    super();

    this.addOutputInterface('Output');
  }
}
