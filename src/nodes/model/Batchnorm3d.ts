import { Node } from '@baklavajs/core';
import { ModelNodes } from '@/nodes/model/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';
import CheckboxValue from '@/baklava/CheckboxValue';

export enum BatchNorm3dOptions {
  NumFeatures = 'Num features',
  Eps = 'Eps',
  Momentum = 'Momentum',
  Affine = 'Affine',
  TrackRunningStats = 'Track running stats'
}
export default class BatchNorm3d extends Node {
  type = ModelNodes.BatchNorm3d;
  name = ModelNodes.BatchNorm3d;

  constructor() {
    super();
    this.addInputInterface('Input');
    this.addOutputInterface('Output');
    this.addOption(BatchNorm3dOptions.NumFeatures, TypeOptions.IntOption, [0, 0, 0]);
    this.addOption(BatchNorm3dOptions.Eps, TypeOptions.SliderOption, [1e-05, 1e-05, 1e-05]);
    this.addOption(BatchNorm3dOptions.Momentum, TypeOptions.SliderOption, [0.1, 0.1, 0.1]);
    this.addOption(BatchNorm3dOptions.Affine, TypeOptions.TickBoxOption, CheckboxValue.CHECKED);
    this.addOption(BatchNorm3dOptions.TrackRunningStats, TypeOptions.TickBoxOption, CheckboxValue.CHECKED);
  }
}
