import { Node } from '@baklavajs/core';
import { Layers, Nodes } from '@/nodes/model/Types';
import { valuesOf } from '@/app/util';
import { BuiltinActivationF, BuiltinInitializer, BuiltinRegularizer } from '@/app/ir/irCommon';
import CheckboxValue from '@/baklava/CheckboxValue';

export default class Dense extends Node {
  type = Layers.Linear;
  name = Nodes.Dense;

  constructor() {
    super();
    this.addInputInterface('Input');
    this.addOutputInterface('Output');

    this.addOption('Size', 'IntOption', 1);
    this.addOption('Activation', 'DropdownOption', 'None', undefined, {
      items: valuesOf(BuiltinActivationF),
    });
    this.addOption('Use Bias', 'TickBoxOption', CheckboxValue.CHECKED);

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
