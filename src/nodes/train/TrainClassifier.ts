import { Node } from '@baklavajs/core';
import { valuesOf } from '@/app/util';
import { OverviewNodes } from '@/nodes/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';

export enum TrainClassifierOptions {
  LossFunction = 'LossFunction',
  Epochs = 'Epochs',
  Device = 'Device',
  LogInterval = 'Log Interval'
}

export enum TrainClassifierLossFunctions {
  CrossEntropyLoss = 'cross_entropy',
  NllLoss = 'nll_loss',
}

export enum Devices {
  CPU = 'cpu',
  CUDA = 'cuda',
}

export default class TrainClassifier extends Node {
  type = OverviewNodes.TrainClassifier;
  name = OverviewNodes.TrainClassifier;

  constructor() {
    super();

    this.addOption(TrainClassifierOptions.LossFunction, 'DropdownOption', TrainClassifierLossFunctions.CrossEntropyLoss, undefined, {
      items: valuesOf(TrainClassifierLossFunctions),
    });
    this.addOption(TrainClassifierOptions.Epochs, TypeOptions.IntOption, 10);
    this.addOption(TrainClassifierOptions.Device, TypeOptions.DropdownOption, Devices.CPU,
      undefined,
      {
        items: valuesOf(Devices),
      });
    this.addOption(TrainClassifierOptions.LogInterval, TypeOptions.IntOption, 0);

    this.addInputInterface('Predictions');
    this.addInputInterface('Labels');
    this.addInputInterface('Optimizer');
  }
}
