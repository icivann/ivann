import { Node } from '@baklavajs/core';
import { ModelNodes } from '@/nodes/model/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';
import CheckboxValue from '@/baklava/CheckboxValue';

export enum AdaptiveAvgPool3dOptions {
  OutputSize = 'Output size'
}
export default class AdaptiveAvgPool3d extends Node {
  type = ModelNodes.AdaptiveAvgPool3d;
  name = ModelNodes.AdaptiveAvgPool3d;

  constructor() {
    super();
    this.addInputInterface('Input');
    this.addOutputInterface('Output');
    this.addOption(AdaptiveAvgPool3dOptions.OutputSize, TypeOptions.VectorOption, [0, 0, 0]);
  }
}
