import { Node } from '@baklavajs/core';
import { Layers, Nodes } from '@/nodes/model/Types';
import {
  BuiltinActivationF,
  BuiltinInitializer,
  BuiltinRegularizer,
  Padding,
} from '@/app/ir/irCommon';
import ModelNode from '@/app/ir/Conv2D';
import GraphNode from '@/app/ir/GraphNode';
import { randomUuid } from '@/app/util';

export default class Conv2D extends Node {
  type = Layers.Conv;
  name = Nodes.Conv2D;

  constructor() {
    super();
    this.addInputInterface('Input');
    this.addOutputInterface('Output');

    this.addOption('Filters', 'IntegerOption');

    // TODO: Keras+Pytorch allow shortcut for specifying single int for all dimensions
    this.addOption('Kernel Size Height', 'IntegerOption');
    this.addOption('Kernel Size Width', 'IntegerOption');
    this.addOption('Stride Height', 'IntegerOption');
    this.addOption('Stride Width', 'IntegerOption');

    this.addOption('Padding', 'SelectOption', 'Valid', undefined, {
      items: ['Valid', 'Same'],
    });
    this.addOption('Activation', 'SelectOption', 'None', undefined, {
      items: ['Relu'],
    });
    this.addOption('Use Bias', 'CheckboxOption', true);

    // TODO: Decide default value and options for these
    this.addOption('Weights Initializer', 'SelectOption', 'Xavier', undefined, {
      items: ['Zeros', 'Glorot_Uniform'],
    });
    this.addOption('Bias Initializer', 'SelectOption', 'Zeros', undefined, {
      items: ['Zeros', 'Glorot_Uniform'],
    });
    this.addOption('Bias Regularizer', 'SelectOption', 'None', undefined, {
      items: ['None'],
    });
    this.addOption('Weights Regularizer', 'SelectOption', 'None', undefined, {
      items: ['None'],
    });
  }

  public calculate() {
    const filters = this.getOptionValue('Filters') as bigint;

    const kernel_h = this.getOptionValue('Kernel Size Height');
    const kernel_w = this.getOptionValue('Kernel Size Width');
    const stride_h = this.getOptionValue('Stride Height');
    const stride_w = this.getOptionValue('Stride Width');

    const padding: Padding = Padding[this.getOptionValue('Padding') as keyof typeof Padding];

    const activation = BuiltinActivationF[this.getOptionValue('Activation') as keyof typeof BuiltinActivationF];

    // const use_bias = this.getOptionValue('Use Bias');

    // TODO: Decide default value and options for these
    const weights_initializer = BuiltinInitializer[this.getOptionValue('Weights Initializer') as keyof typeof BuiltinInitializer];
    const weights_regularizer = BuiltinRegularizer[this.getOptionValue('Weights Regularizer') as keyof typeof BuiltinRegularizer];
    const bias_initializer = BuiltinInitializer[this.getOptionValue('Bias Initializer') as keyof typeof BuiltinInitializer];
    const bias_regularizer = BuiltinRegularizer[this.getOptionValue('Bias Regularizer') as keyof typeof BuiltinRegularizer];

    const layer = new ModelNode(
      new Set(),
      filters,
      padding,
      [weights_initializer, weights_regularizer],
      [bias_initializer, weights_regularizer],
      randomUuid(),
      activation,
      [kernel_h, kernel_w],
      [stride_h, stride_w],
    );

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
