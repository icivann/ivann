import { Node } from '@baklavajs/core';
import { Nodes } from '@/nodes/model/Types';

export default class Output extends Node {
  type = Nodes.Output;
  name = Nodes.Output;

  constructor() {
    super();
    this.addInputInterface('Input');
  }
}
