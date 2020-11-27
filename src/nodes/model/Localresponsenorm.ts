import { Node } from '@baklavajs/core';
import { ModelNodes } from '@/nodes/model/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';
import CheckboxValue from '@/baklava/CheckboxValue';

export enum LocalResponseNormOptions {
  Size = 'Size',
  Alpha = 'Alpha',
  Beta = 'Beta',
  K = 'K'
}
export default class LocalResponseNorm extends Node {
  type = ModelNodes.LocalResponseNorm;
  name = ModelNodes.LocalResponseNorm;

  constructor() {
    super();
    this.addInputInterface('Input');
    this.addOutputInterface('Output');
    this.addOption(LocalResponseNormOptions.Size, TypeOptions.IntOption, 0);
    this.addOption(LocalResponseNormOptions.Alpha, TypeOptions.SliderOption, 0.0001);
    this.addOption(LocalResponseNormOptions.Beta, TypeOptions.SliderOption, 0.75);
    this.addOption(LocalResponseNormOptions.K, TypeOptions.SliderOption, 1.0);
  }
}
