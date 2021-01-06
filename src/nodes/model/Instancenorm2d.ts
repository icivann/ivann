import { Node } from '@baklavajs/core';
import { ModelNodes } from '@/nodes/model/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';
import CheckboxValue from '@/baklava/CheckboxValue';

export enum InstanceNorm2dOptions {
  NumFeatures = 'Num features',
  Eps = 'Eps',
  Momentum = 'Momentum',
  Affine = 'Affine',
  TrackRunningStats = 'Track running stats'
}
export default class InstanceNorm2d extends Node {
  type = ModelNodes.InstanceNorm2d;
  name = ModelNodes.InstanceNorm2d;

  constructor() {
    super();
    this.addInputInterface('Input');
    this.addOutputInterface('Output');
    this.addOption(InstanceNorm2dOptions.NumFeatures, TypeOptions.IntOption, [0, 0]);
    this.addOption(InstanceNorm2dOptions.Eps, TypeOptions.SliderOption, [1e-05, 1e-05]);
    this.addOption(InstanceNorm2dOptions.Momentum, TypeOptions.SliderOption, [0.1, 0.1]);
    this.addOption(InstanceNorm2dOptions.Affine, TypeOptions.TickBoxOption, CheckboxValue.UNCHECKED);
    this.addOption(InstanceNorm2dOptions.TrackRunningStats, TypeOptions.TickBoxOption, CheckboxValue.UNCHECKED);
  }
}
