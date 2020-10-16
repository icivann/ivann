import { Node } from '@baklavajs/core';
import { Layers, Nodes } from '@/nodes/model/Types';
import ModelNode from '@/app/ir/InModel';
import GraphNode from '@/app/ir/GraphNode';

export default class InModel extends Node {
  type = Layers.IO;
  name = Nodes.InModel;

  constructor() {
    super();
    this.addOutputInterface('Output');

    this.addOption('dim 0', 'IntegerOption');
    this.addOption('dim 1', 'IntegerOption');
    this.addOption('dim 2', 'IntegerOption');
  }

  public calculate() {
    const dim_0 = this.getOptionValue('dim 0') as bigint;
    const dim_1 = this.getOptionValue('dim 1')as bigint;
    const dim_2 = this.getOptionValue('dim 2')as bigint;

    const layer = new ModelNode(new Set(), [dim_0, dim_1, dim_2]);
    const graph_node = new GraphNode(layer);

    this.getInterface('Output').value = [graph_node];
  }
}
