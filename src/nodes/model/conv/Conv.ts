import { Node } from '@baklavajs/core';
import {
  BuiltinActivationF,
  BuiltinInitializer,
  BuiltinRegularizer,
  Padding,
} from '@/app/ir/irCommon';
import { valuesOf } from '@/app/util';
import CheckboxValue from '@/baklava/CheckboxValue';

export enum ConvOptions {
  Filters = 'Filters', KernelSize = 'Kernel Size', Stride = 'Stride',
  Padding = 'Padding', Activation = 'Activation',
  UseBias = 'Use Bias', WeightsInitializer = 'Weights Initializer',
  BiasInitializer = 'Bias Initializer', WeightsRegularizer = 'Weights Regularizer',
  BiasRegularizer = 'Bias Regularizer'
}

export default abstract class Conv extends Node {
  constructor() {
    super();
    this.addInputInterface('Input');
    this.addOutputInterface('Output');

    this.addOption(ConvOptions.Filters, 'IntOption', 1, undefined, {
      min: 1,
    });

    this.addKernelStride();

    this.addOption(ConvOptions.Padding, 'DropdownOption', 'Valid', undefined, {
      items: valuesOf(Padding),
    });
    this.addOption(ConvOptions.Activation, 'DropdownOption', 'None', undefined, {
      items: valuesOf(BuiltinActivationF),
    });
    this.addOption(ConvOptions.UseBias, 'TickBoxOption', CheckboxValue.CHECKED);

    // TODO: Decide default value and options for these
    this.addOption(ConvOptions.WeightsInitializer, 'DropdownOption', 'Xavier', undefined, {
      items: valuesOf(BuiltinInitializer),
    });
    this.addOption(ConvOptions.BiasInitializer, 'DropdownOption', 'Zeros', undefined, {
      items: valuesOf(BuiltinInitializer),
    });
    this.addOption(ConvOptions.BiasRegularizer, 'DropdownOption', 'None', undefined, {
      items: valuesOf(BuiltinRegularizer),
    });
    this.addOption(ConvOptions.WeightsRegularizer, 'DropdownOption', 'None', undefined, {
      items: valuesOf(BuiltinRegularizer),
    });
  }

  protected abstract addKernelStride(): void;
}
