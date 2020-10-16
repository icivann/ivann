import { Node } from '@baklavajs/core';
import { Layers, Nodes } from '@/nodes/model/Types';
import ModelNode from '@/app/ir/dropout';
import GraphNode from '@/app/ir/GraphNode';
import { randomUuid } from '@/app/util';

export default class Dropout extends Node {
  type = Layers.Regularization
  name = Nodes.Dropout;

  constructor() {
    super();
    this.addInputInterface('Input');
    this.addOutputInterface('Output');

    this.addOption('Probability', 'SliderOption', 0.5, undefined, {
      min: 0,
      max: 1,
    });
  }

  public calculate() {
    const p = this.getOptionValue('Probability');


    const layer = new ModelNode(randomUuid(), new Set(), p);

    const data = this.getInterface('Input').value as GraphNode[];
    const graph_node = new GraphNode(layer);
    console.log(data, typeof data);
    if (data == null) {
      this.getInterface('Output').value = [graph_node];
    } else {
      this.getInterface('Output').value = data.concat([graph_node]);
    }
  }
}
