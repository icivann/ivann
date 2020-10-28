import { Node } from '@baklavajs/core';
import { Nodes } from '@/nodes/train/Types';
import { valuesOf } from '@/app/util';

export enum ToDeviceOptions {
  Device = 'Device',
}

export enum Devices {
  CPU = 'CPU', GPU='GPU',
}

export default class ToDevice extends Node {
  type = Nodes.DatasetInput;
  name = Nodes.DatasetInput;

  constructor() {
    super();
    this.addInputInterface('in');
    this.addOutputInterface('out');
    this.addOption(ToDeviceOptions.Device, 'DropdownOption', 'Valid', undefined, {
      items: valuesOf(Devices),
    });
  }
}
