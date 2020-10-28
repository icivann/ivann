import { Node } from '@baklavajs/core';
import { Nodes } from '@/nodes/train/Types';

export default class Backpropagate extends Node {
  type = Nodes.Backpropagate;
  name = Nodes.Backpropagate;

  constructor() {
    super();
    this.addInputInterface('loss');
  }
}
