import { Node } from '@baklavajs/core';
import {
  BuiltinActivationF,
  BuiltinInitializer,
  BuiltinRegularizer,
  Padding,
} from '@/app/ir/irCommon';
import { valuesOf } from '@/app/util';

export default abstract class Conv extends Node {
  constructor() {
    super();
    this.addInputInterface('Input');
    this.addOutputInterface('Output');

    this.addOption('Filters', 'IntOption', 1, undefined, {
      min: 1,
    });

    this.addKernelStride();

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

  protected abstract addKernelStride(): void;
}
