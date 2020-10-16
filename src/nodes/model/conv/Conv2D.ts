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

    this.addOption('Filters', 'IntegerOption', 64);

    // TODO: Keras+Pytorch allow shortcut for specifying single int for all dimensions
    this.addOption('Kernel Size Height', 'IntegerOption', 3);
    this.addOption('Kernel Size Width', 'IntegerOption', 3);
    this.addOption('Stride Height', 'IntegerOption', 1);
    this.addOption('Stride Width', 'IntegerOption', 1);

    this.addOption('Padding', 'SelectOption', 'Valid', undefined, {
      items: ['Valid', 'Same'],
    });
    this.addOption('Activation', 'SelectOption', 'Relu', undefined, {
      items: ['Relu'],
    });
    this.addOption('Use Bias', 'CheckboxOption', true);

    // TODO: Decide default value and options for these
    this.addOption('Weights Initializer', 'SelectOption', 'Glorot_Uniform', undefined, {
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

    const kernelH = this.getOptionValue('Kernel Size Height');
    const kernelW = this.getOptionValue('Kernel Size Width');
    const strideH = this.getOptionValue('Stride Height');
    const strideW = this.getOptionValue('Stride Width');

    const padding: Padding = Padding[this.getOptionValue('Padding') as keyof typeof Padding];

    const activation = BuiltinActivationF[this.getOptionValue('Activation') as keyof typeof BuiltinActivationF];

    // const use_bias = this.getOptionValue('Use Bias');

    // TODO: Decide default value and options for these
    const weightsInitializer = BuiltinInitializer[this.getOptionValue('Weights Initializer') as keyof typeof BuiltinInitializer];
    const weightsRegularizer = BuiltinRegularizer[this.getOptionValue('Weights Regularizer') as keyof typeof BuiltinRegularizer];
    const biasInitializer = BuiltinInitializer[this.getOptionValue('Bias Initializer') as keyof typeof BuiltinInitializer];
    const biasRegularizer = BuiltinRegularizer[this.getOptionValue('Bias Regularizer') as keyof typeof BuiltinRegularizer];

    const layer = new ModelNode(
      new Set(),
      filters,
      padding,
      [weightsInitializer, weightsRegularizer],
      [biasInitializer, weightsRegularizer],
      randomUuid(),
      activation,
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
