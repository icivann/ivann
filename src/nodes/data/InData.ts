import { Node } from '@baklavajs/core';
import { Nodes, NodeTypes } from '@/nodes/data/Types';

export default class InData extends Node {
  type = NodeTypes.IO;
  name = Nodes.InData;

  constructor() {
    super();

    this.addOutputInterface('Output');
  }
}
