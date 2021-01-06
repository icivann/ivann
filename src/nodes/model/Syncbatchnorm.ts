import { Node } from '@baklavajs/core';
import { ModelNodes } from '@/nodes/model/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';
import CheckboxValue from '@/baklava/CheckboxValue';

export enum SyncBatchNormOptions {
  NumFeatures = 'Num features',
  Eps = 'Eps',
  Momentum = 'Momentum',
  Affine = 'Affine',
  TrackRunningStats = 'Track running stats',
  ProcessGroup = 'Process group'
}
export default class SyncBatchNorm extends Node {
  type = ModelNodes.SyncBatchNorm;
  name = ModelNodes.SyncBatchNorm;

  constructor() {
    super();
    this.addInputInterface('Input');
    this.addOutputInterface('Output');
    this.addOption(SyncBatchNormOptions.NumFeatures, TypeOptions.IntOption, 0);
    this.addOption(SyncBatchNormOptions.Eps, TypeOptions.SliderOption, 1e-05);
    this.addOption(SyncBatchNormOptions.Momentum, TypeOptions.SliderOption, 0.1);
    this.addOption(SyncBatchNormOptions.Affine, TypeOptions.TickBoxOption, CheckboxValue.CHECKED);
    this.addOption(SyncBatchNormOptions.TrackRunningStats, TypeOptions.TickBoxOption, CheckboxValue.CHECKED);
    this.addOption(SyncBatchNormOptions.ProcessGroup, TypeOptions.VectorOption, [0]);
  }
}
