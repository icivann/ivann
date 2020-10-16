import { Node } from '@baklavajs/core';
import { Layers, Nodes } from '@/nodes/model/Types';
import {
  BuiltinActivationF,
  BuiltinInitializer,
  BuiltinRegularizer,
  Padding,
} from '@/app/ir/irCommon';
import ModelNode from '@/app/ir/MaxPool2D';
import GraphNode from '@/app/ir/GraphNode';
import { randomUuid } from '@/app/util';

export default class MaxPool2D extends Node {
  type = Layers.Pool;
  name = Nodes.MaxPool2D;

  constructor() {
    super();
    this.addInputInterface('Input');
    this.addOutputInterface('Output');

    // TODO: Keras+Pytorch allow shortcut for specifying single int for all dimensions
    this.addOption('Kernel Size Height', 'IntegerOption', 3);
    this.addOption('Kernel Size Width', 'IntegerOption', 3);
    this.addOption('Stride Height', 'IntegerOption', 1);
    this.addOption('Stride Width', 'IntegerOption', 1);

    this.addOption('Padding', 'SelectOption', 'Valid', undefined, {

      items: ['Valid', 'Same'],
    });
  }

  public calculate() {
    const kernelH = this.getOptionValue('Kernel Size Height');
    const kernelW = this.getOptionValue('Kernel Size Width');
    const strideH = this.getOptionValue('Stride Height');
    const strideW = this.getOptionValue('Stride Width');

    const padding: Padding = Padding[this.getOptionValue('Padding') as keyof typeof Padding];

    const layer = new ModelNode(
      new Set(), randomUuid(), padding,
      [kernelH, kernelW],
      [strideH, strideW],
    );

    const data = this.getInterface('Input').value as GraphNode[];
    const graphNode = new GraphNode(layer);
    console.log(data, typeof data);
    if (data == null) {
      this.getInterface('Output').value = [graphNode];
    } else {
      this.getInterface('Output').value = data.concat([graphNode]);
    }
  }
}
