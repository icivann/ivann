import { Node } from '@baklavajs/core';
import { ModelNodes } from '@/nodes/model/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';
import CheckboxValue from '@/baklava/CheckboxValue';

export enum BatchNorm1dOptions {
  NumFeatures = 'Num features',
  Eps = 'Eps',
  Momentum = 'Momentum',
  Affine = 'Affine',
  TrackRunningStats = 'Track running stats'
}
export default class BatchNorm1d extends Node {
  type = ModelNodes.BatchNorm1d;
  name = ModelNodes.BatchNorm1d;

  constructor() {
    super();
    this.addInputInterface('Input');
    this.addOutputInterface('Output');
    this.addOption(BatchNorm1dOptions.NumFeatures, TypeOptions.IntOption, [0]);
    this.addOption(BatchNorm1dOptions.Eps, TypeOptions.SliderOption, [1e-05]);
    this.addOption(BatchNorm1dOptions.Momentum, TypeOptions.SliderOption, [0.1]);
    this.addOption(BatchNorm1dOptions.Affine, TypeOptions.TickBoxOption, CheckboxValue.CHECKED);
    this.addOption(BatchNorm1dOptions.TrackRunningStats, TypeOptions.TickBoxOption, CheckboxValue.CHECKED);
  }
}
