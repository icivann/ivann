import { Node } from '@baklavajs/core';
import { Nodes, NodeTypes } from '@/nodes/data/Types';

export enum OutDataOptions {
  BatchSize = 'Batch Size',
}

class OutData extends Node {
  type = NodeTypes.IO;
  name = Nodes.OutData;

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
