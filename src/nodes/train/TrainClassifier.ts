import { Node } from '@baklavajs/core';
import { valuesOf } from '@/app/util';
import { Nodes, OverviewNodes } from '@/nodes/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';

export enum TrainClassifierOptions {
  LossFunction = 'LossFunction',
  Optimizer = 'Optimizer',
  BatchSize = 'BatchSize',
  Epochs = 'Epochs',
}

export enum TrainClassifierLossFunctions {
  CrossEntropyLoss = 'Cross Entropy Loss',
}

export default class TrainClassifier extends Node {
  type = OverviewNodes.TrainClassifier;
  name = OverviewNodes.TrainClassifier;

  constructor() {
    super();

    this.addOption(TrainClassifierOptions.LossFunction, TypeOptions.DropdownOption);
    this.addOption(TrainClassifierOptions.BatchSize, TypeOptions.IntOption, 32);
    this.addOption(TrainClassifierOptions.Epochs, TypeOptions.IntOption, 100);

    this.addOption('Activation', 'DropdownOption', 'None', undefined, {
      items: valuesOf(TrainClassifierLossFunctions),
    });

    this.addInputInterface('Predictions');
    this.addInputInterface('Labels');
    this.addInputInterface('Optimizer');
  }
}
