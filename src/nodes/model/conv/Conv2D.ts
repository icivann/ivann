import { Node } from '@baklavajs/core';
import { Layers, Nodes } from '@/nodes/model/Types';
import {
  BuiltinActivationF,
  BuiltinInitializer,
  BuiltinRegularizer,
  Padding,
} from '@/app/ir/irCommon';
import { valuesOf } from '@/app/util';

export enum Conv2DOptions{ Filters= 'Filters', KernelSize='Kernel Size', Stride='Stride',
Padding ='Padding', Activation = 'Activation',
UseBias= 'Use Bias', WeightsInitializer = 'Weights Initializer',
BiasInitializer='Bias Initializer', WeightsRegularizer = 'Weights Regularizer',
BiasRegularizer='Bias Regularizer'
}
export default class Conv2D extends Node {
  type = Layers.Conv;
  name = Nodes.Conv2D;

  constructor() {
    super();
    this.addInputInterface('Input');
    this.addOutputInterface('Output');


    this.addOption(Conv2DOptions.Filters, 'IntOption', 1,  undefined, {
      min: 1,
    });

    // TODO: Keras+Pytorch allow shortcut for specifying single int for all dimensions
    this.addOption(Conv2DOptions.KernelSize, 'VectorOption', [1, 1], undefined, {
      min: [1, 1],
    });
    this.addOption(Conv2DOptions.Stride, 'VectorOption', [1, 1], undefined, {
      min: [1, 1],
    });

    this.addOption(Conv2DOptions.Padding, 'DropdownOption', 'Valid', undefined, {
      items: valuesOf(Padding),
    });
    this.addOption(Conv2DOptions.Activation, 'DropdownOption', 'None', undefined, {
      items: valuesOf(BuiltinActivationF),
    });
    this.addOption(Conv2DOptions.UseBias, 'CheckboxOption', true);

    // TODO: Decide default value and options for these
    this.addOption(Conv2DOptions.WeightsInitializer, 'DropdownOption', 'Xavier', undefined, {
      items: valuesOf(BuiltinInitializer),
    });
    this.addOption(Conv2DOptions.BiasInitializer, 'DropdownOption', 'Zeroes', undefined, {
      items: valuesOf(BuiltinInitializer),
    });
    this.addOption(Conv2DOptions.BiasRegularizer, 'DropdownOption', 'None', undefined, {
      items: valuesOf(BuiltinRegularizer),
    });
    this.addOption(Conv2DOptions.WeightsRegularizer, 'DropdownOption', 'None', undefined, {
      items: valuesOf(BuiltinRegularizer),
    });
  }
}
