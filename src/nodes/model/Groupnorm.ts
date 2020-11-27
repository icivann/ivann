import { Node } from '@baklavajs/core';
import { ModelNodes } from '@/nodes/model/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';
import CheckboxValue from '@/baklava/CheckboxValue';

export enum GroupNormOptions {
  NumGroups = 'Num groups',
  NumChannels = 'Num channels',
  Eps = 'Eps',
  Affine = 'Affine'
}
export default class GroupNorm extends Node {
  type = ModelNodes.GroupNorm;
  name = ModelNodes.GroupNorm;

  constructor() {
    super();
    this.addInputInterface('Input');
    this.addOutputInterface('Output');
    this.addOption(GroupNormOptions.NumGroups, TypeOptions.IntOption, 0);
    this.addOption(GroupNormOptions.NumChannels, TypeOptions.IntOption, 0);
    this.addOption(GroupNormOptions.Eps, TypeOptions.SliderOption, 1e-05);
    this.addOption(GroupNormOptions.Affine, TypeOptions.TickBoxOption, CheckboxValue.CHECKED);
  }
}
