import { Node } from '@baklavajs/core';
import { DataNodes } from '@/nodes/data/Types';

export enum OutDataOptions {
  BatchSize = 'Batch Size',
}

class OutData extends Node {
  type = DataNodes.OutData;
  name = DataNodes.OutData;

  constructor() {
    super();

    this.addInputInterface('Input');
    this.addInputInterface('Label');

    this.addOption(OutDataOptions.BatchSize, 'IntOption', 1, undefined, {
      min: 1,
    });
  }
}

export default OutData;
