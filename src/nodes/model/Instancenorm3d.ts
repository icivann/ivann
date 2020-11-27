import { Node } from '@baklavajs/core';
import { ModelNodes } from '@/nodes/model/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';
import CheckboxValue from '@/baklava/CheckboxValue';

export enum InstanceNorm3dOptions {
  NumFeatures = 'Num features',
  Eps = 'Eps',
  Momentum = 'Momentum',
  Affine = 'Affine',
  TrackRunningStats = 'Track running stats'
}
export default class InstanceNorm3d extends Node {
  type = ModelNodes.InstanceNorm3d;
  name = ModelNodes.InstanceNorm3d;

  constructor() {
    super();
    this.addInputInterface('Input');
    this.addOutputInterface('Output');
    this.addOption(InstanceNorm3dOptions.NumFeatures, TypeOptions.IntOption, [0, 0, 0]);
    this.addOption(InstanceNorm3dOptions.Eps, TypeOptions.SliderOption, [1e-05, 1e-05, 1e-05]);
    this.addOption(InstanceNorm3dOptions.Momentum, TypeOptions.SliderOption, [0.1, 0.1, 0.1]);
    this.addOption(InstanceNorm3dOptions.Affine, TypeOptions.TickBoxOption, CheckboxValue.UNCHECKED);
    this.addOption(InstanceNorm3dOptions.TrackRunningStats, TypeOptions.TickBoxOption, CheckboxValue.UNCHECKED);
  }
}
