import { Node } from '@baklavajs/core';
import { ModelNodes } from '@/nodes/model/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';
import CheckboxValue from '@/baklava/CheckboxValue';

export enum AdaptiveMaxPool3dOptions {
  OutputSize = 'Output size',
  ReturnIndices = 'Return indices'
}
export default class AdaptiveMaxPool3d extends Node {
  type = ModelNodes.AdaptiveMaxPool3d;
  name = ModelNodes.AdaptiveMaxPool3d;

  constructor() {
    super();
    this.addInputInterface('Input');
    this.addOutputInterface('Output');
    this.addOption(AdaptiveMaxPool3dOptions.OutputSize, TypeOptions.VectorOption, [0, 0, 0]);
    this.addOption(AdaptiveMaxPool3dOptions.ReturnIndices, TypeOptions.TickBoxOption, CheckboxValue.UNCHECKED);
  }
}
