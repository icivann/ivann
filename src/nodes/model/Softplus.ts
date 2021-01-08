import { Node } from '@baklavajs/core';
import { ModelNodes } from '@/nodes/model/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';
import CheckboxValue from '@/baklava/CheckboxValue';

export enum SoftplusOptions {
  Beta = 'Beta',
  Threshold = 'Threshold'
}
export default class Softplus extends Node {
  type = ModelNodes.Softplus;
  name = ModelNodes.Softplus;

  constructor() {
    super();
    this.addInputInterface('Input');
    this.addOutputInterface('Output');
    this.addOption(SoftplusOptions.Beta, TypeOptions.IntOption, 1);
    this.addOption(SoftplusOptions.Threshold, TypeOptions.IntOption, 20);
  }
}
