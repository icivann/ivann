import { Node } from '@baklavajs/core';
import { Nodes } from '@/nodes/model/Types';
import { valuesOf } from '@/app/util';
import { BuiltinActivationF, BuiltinInitializer, BuiltinRegularizer } from '@/app/ir/irCommon';
import CheckboxValue from '@/baklava/CheckboxValue';

export enum DenseOptions {
  Size = 'Size', Activation = 'Activation', UseBias= 'Use Bias',
  WeightsInitializer = 'Weights Initializer', BiasInitializer = 'Bias Initializer',
  BiasRegularizer = 'Bias Regularizer', WeightsRegularizer = 'Weights Regularizer'
}
export default class Dense extends Node {
  type = Nodes.Dense;
  name = Nodes.Dense;

  constructor() {
    super();
    this.addInputInterface('Input');
    this.addOutputInterface('Output');

    this.addOption(DenseOptions.Size, 'IntOption', 1, undefined, {
      min: 1,
    });
    this.addOption(DenseOptions.Activation, 'DropdownOption', 'None', undefined, {
      items: valuesOf(BuiltinActivationF),
    });
    this.addOption(DenseOptions.UseBias, 'TickBoxOption', CheckboxValue.CHECKED);

    // TODO: Decide default value and options for these
    this.addOption(DenseOptions.WeightsInitializer, 'DropdownOption', 'Xavier', undefined, {
      items: valuesOf(BuiltinInitializer),
    });
    this.addOption(DenseOptions.BiasInitializer, 'DropdownOption', 'Zeroes', undefined, {
      items: valuesOf(BuiltinInitializer),
    });
    this.addOption(DenseOptions.BiasRegularizer, 'DropdownOption', 'None', undefined, {
      items: valuesOf(BuiltinRegularizer),
    });
    this.addOption(DenseOptions.WeightsRegularizer, 'DropdownOption', 'None', undefined, {
      items: valuesOf(BuiltinRegularizer),
    });
  }
}
