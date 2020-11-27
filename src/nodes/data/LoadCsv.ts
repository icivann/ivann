import { Node } from '@baklavajs/core';
import { DataNodes } from '@/nodes/data/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';

export enum LoadCsvOptions {
  Column = 'Column',
}

export default class LoadCsv extends Node {
  type = DataNodes.LoadCsv;
  name = DataNodes.LoadCsv;

  constructor() {
    super();

    this.addOption(LoadCsvOptions.Column, TypeOptions.IntOption, 0);
    this.addOutputInterface('Output');
  }
}
