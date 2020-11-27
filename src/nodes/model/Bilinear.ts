import { Node } from '@baklavajs/core';
import { ModelNodes } from '@/nodes/model/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';
import CheckboxValue from '@/baklava/CheckboxValue';

export enum BilinearOptions {
  In1Features = 'In1 features',
  In2Features = 'In2 features',
  OutFeatures = 'Out features',
  Bias = 'Bias'
}
export default class Bilinear extends Node {
  type = ModelNodes.Bilinear;
  name = ModelNodes.Bilinear;

  constructor() {
    super();
    this.addInputInterface('Input1');
    this.addInputInterface('Input2');
    this.addOutputInterface('Output');
    this.addOption(BilinearOptions.In1Features, TypeOptions.IntOption, 0);
    this.addOption(BilinearOptions.In2Features, TypeOptions.IntOption, 0);
    this.addOption(BilinearOptions.OutFeatures, TypeOptions.IntOption, 0);
    this.addOption(BilinearOptions.Bias, TypeOptions.TickBoxOption, CheckboxValue.CHECKED);
  }
}
