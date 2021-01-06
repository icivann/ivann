import { Node } from '@baklavajs/core';
import { ModelNodes } from '@/nodes/model/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';
import CheckboxValue from '@/baklava/CheckboxValue';

export enum HardtanhOptions {
  MinVal = 'Min val',
  MaxVal = 'Max val',
  Inplace = 'Inplace',
  MinValue = 'Min value',
  MaxValue = 'Max value'
}
export default class Hardtanh extends Node {
  type = ModelNodes.Hardtanh;
  name = ModelNodes.Hardtanh;

  constructor() {
    super();
    this.addInputInterface('Input');
    this.addOutputInterface('Output');
    this.addOption(HardtanhOptions.MinVal, TypeOptions.SliderOption, -1.0);
    this.addOption(HardtanhOptions.MaxVal, TypeOptions.SliderOption, 1.0);
    this.addOption(HardtanhOptions.Inplace, TypeOptions.TickBoxOption, CheckboxValue.UNCHECKED);
    this.addOption(HardtanhOptions.MinValue, TypeOptions.VectorOption, null);
    this.addOption(HardtanhOptions.MaxValue, TypeOptions.VectorOption, null);
  }
}
