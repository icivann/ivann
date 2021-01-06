import { Node } from '@baklavajs/core';
import { ModelNodes } from '@/nodes/model/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';
import CheckboxValue from '@/baklava/CheckboxValue';

export enum AdaptiveAvgPool2dOptions {
  OutputSize = 'Output size'
}
export default class AdaptiveAvgPool2d extends Node {
  type = ModelNodes.AdaptiveAvgPool2d;
  name = ModelNodes.AdaptiveAvgPool2d;

  constructor() {
    super();
    this.addInputInterface('Input');
    this.addOutputInterface('Output');
    this.addOption(AdaptiveAvgPool2dOptions.OutputSize, TypeOptions.VectorOption, [0, 0]);
  }
}
