import { Node } from '@baklavajs/core';
import { ModelNodes } from '@/nodes/model/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';
import CheckboxValue from '@/baklava/CheckboxValue';

export enum InstanceNorm1dOptions {
  NumFeatures = 'Num features',
  Eps = 'Eps',
  Momentum = 'Momentum',
  Affine = 'Affine',
  TrackRunningStats = 'Track running stats'
}
export default class InstanceNorm1d extends Node {
  type = ModelNodes.InstanceNorm1d;
  name = ModelNodes.InstanceNorm1d;

  constructor() {
    super();
    this.addInputInterface('Input');
    this.addOutputInterface('Output');
    this.addOption(InstanceNorm1dOptions.NumFeatures, TypeOptions.IntOption, [0]);
    this.addOption(InstanceNorm1dOptions.Eps, TypeOptions.SliderOption, [1e-05]);
    this.addOption(InstanceNorm1dOptions.Momentum, TypeOptions.SliderOption, [0.1]);
    this.addOption(InstanceNorm1dOptions.Affine, TypeOptions.TickBoxOption, CheckboxValue.UNCHECKED);
    this.addOption(InstanceNorm1dOptions.TrackRunningStats, TypeOptions.TickBoxOption, CheckboxValue.UNCHECKED);
  }
}
