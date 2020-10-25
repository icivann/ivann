import { Node } from '@baklavajs/core';
import { Layers, Nodes } from '@/nodes/model/Types';
import {
  BuiltinActivationF,
  BuiltinInitializer,
  BuiltinRegularizer,
  Padding,
} from '@/app/ir/irCommon';
import { valuesOf } from '@/app/util';

export default class Conv2D extends Node {
  type = Layers.Conv;
  name = Nodes.Conv2D;

  constructor() {
    super();
    this.addInputInterface('Input');
    this.addOutputInterface('Output');

    this.addOption('Filters', 'IntOption', 1, undefined, {
      min: 1,
    });

    // TODO: Keras+Pytorch allow shortcut for specifying single int for all dimensions
    this.addOption('Kernel Size', 'VectorOption', [1, 1]);
    this.addOption('Stride', 'VectorOption', [1, 1]);

    this.addOption('Padding', 'DropdownOption', 'Valid', undefined, {
      items: valuesOf(Padding),
    });
    this.addOption('Activation', 'DropdownOption', 'None', undefined, {
      items: valuesOf(BuiltinActivationF),
    });
    this.addOption('Use Bias', 'CheckboxOption', true);

    // TODO: Decide default value and options for these
    this.addOption('Weights Initializer', 'DropdownOption', 'Xavier', undefined, {
      items: valuesOf(BuiltinInitializer),
    });
    this.addOption('Bias Initializer', 'DropdownOption', 'Zeros', undefined, {
      items: valuesOf(BuiltinInitializer),
    });
    this.addOption('Bias Regularizer', 'DropdownOption', 'None', undefined, {
      items: valuesOf(BuiltinRegularizer),
    });
    this.addOption('Weights Regularizer', 'DropdownOption', 'None', undefined, {
      items: valuesOf(BuiltinRegularizer),
    });
  }
}
