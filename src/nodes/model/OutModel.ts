import { Node } from '@baklavajs/core';
import { Layers, Nodes } from '@/nodes/model/Types';
import ModelNode from '@/app/ir/OutModel';
import GraphNode from '@/app/ir/GraphNode';
import { randomUuid } from '@/app/util';

export default class OutModel extends Node {
  type = Layers.IO;
  name = Nodes.InModel;

  constructor() {
    super();
    this.addInputInterface('Input');
  }

  public calculate() {
    this.save();
    const data = this.getInterface('Input').value;

    const layer = new ModelNode(randomUuid());
    const graph_node = new GraphNode(layer);
    if (data == null) {
      return [];
    }
    return data;
  }
}
