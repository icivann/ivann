import { Node } from '@baklavajs/core';
import { OverviewNodes } from '@/nodes/overview/Types';

export enum NLLLossOptions {
  Weight = 'Weight',
  IgnoreIndex = 'Ignore index',
  Reduction = 'Reduction'
}

export default class NLLLoss extends Node {
  type = OverviewNodes.Nllloss;
  name = OverviewNodes.Nllloss;

  constructor() {
    super();
    this.addOutputInterface('Output');
  }
}
