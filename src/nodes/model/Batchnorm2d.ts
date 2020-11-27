import { Node } from '@baklavajs/core';
import { ModelNodes } from '@/nodes/model/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';
import CheckboxValue from '@/baklava/CheckboxValue';

export enum BatchNorm2dOptions {
  NumFeatures = 'Num features',
  Eps = 'Eps',
  Momentum = 'Momentum',
  Affine = 'Affine',
  TrackRunningStats = 'Track running stats'
}
export default class BatchNorm2d extends Node {
  type = ModelNodes.BatchNorm2d;
  name = ModelNodes.BatchNorm2d;

  constructor() {
    super();
    this.addInputInterface('Input');
    this.addOutputInterface('Output');
    this.addOption(BatchNorm2dOptions.NumFeatures, TypeOptions.IntOption, [0, 0]);
    this.addOption(BatchNorm2dOptions.Eps, TypeOptions.SliderOption, [1e-05, 1e-05]);
    this.addOption(BatchNorm2dOptions.Momentum, TypeOptions.SliderOption, [0.1, 0.1]);
    this.addOption(BatchNorm2dOptions.Affine, TypeOptions.TickBoxOption, CheckboxValue.CHECKED);
    this.addOption(BatchNorm2dOptions.TrackRunningStats, TypeOptions.TickBoxOption, CheckboxValue.CHECKED);
  }
}
