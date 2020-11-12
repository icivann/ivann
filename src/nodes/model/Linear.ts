import { Node } from '@baklavajs/core';
import { Layers, Nodes } from '@/nodes/model/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';
import CheckboxValue from '@/baklava/CheckboxValue';

export enum LinearOptions {
  InFeatures = 'In features',
  OutFeatures = 'Out features',
  Bias = 'Bias'
}
export default class Linear extends Node {
  type = Nodes.Linear;
  name = Nodes.Linear;

  constructor() {
    super();
    this.addInputInterface('Input');
    this.addOutputInterface('Output');
    this.addOption(LinearOptions.InFeatures, TypeOptions.IntOption, 0);
    this.addOption(LinearOptions.OutFeatures, TypeOptions.IntOption, 0);
    this.addOption(LinearOptions.Bias, TypeOptions.TickBoxOption, CheckboxValue.CHECKED);
  }
}
