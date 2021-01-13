import { Node } from '@baklavajs/core';
import { valuesOf } from '@/app/util';
import { OverviewNodes } from '@/nodes/overview/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';

export enum TrainGANOptions {
  Epochs = 'Epochs',
  Device = 'Device',
  LogInterval = 'Log Interval',
  RealLabel = 'Real Label',
  FakeLabel = 'Fake Label'
}

export enum Devices {
  CPU = 'cpu',
  CUDA = 'cuda',
}

export default class TrainGAN extends Node {
  type = OverviewNodes.TrainGAN;
  name = OverviewNodes.TrainGAN;

  constructor() {
    super();

    this.addOption(TrainGANOptions.Epochs, TypeOptions.IntOption, 10);
    this.addOption(TrainGANOptions.Device, TypeOptions.DropdownOption, Devices.CPU,
      undefined,
      {
        items: valuesOf(Devices),
      });
    this.addOption(TrainGANOptions.LogInterval, TypeOptions.IntOption, 0);
    this.addOption(TrainGANOptions.RealLabel, TypeOptions.IntOption, 1);
    this.addOption(TrainGANOptions.FakeLabel, TypeOptions.IntOption, 0);

    this.addInputInterface('ModelD');
    this.addInputInterface('ModelG');
    this.addInputInterface('Train Data');
    this.addInputInterface('Test Data');
    this.addInputInterface('OptimizerG');
    this.addInputInterface('OptimizerD');
    this.addInputInterface('Loss');
  }
}
