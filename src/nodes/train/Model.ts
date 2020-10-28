import { Node } from '@baklavajs/core';
import { Nodes } from '@/nodes/train/Types';

export enum ModelOptions {
}

export default class Model extends Node {
  type = Nodes.Model;
  name = Nodes.Model;

  private numberOfInputs = 0;
  private numberOfOutputs = 0;

  constructor() {
    super();

    this.addOption('Number of Inputs', 'IntegerOption', 0);
    this.addOption('Number of Outputs', 'IntegerOption', 0);
  }

  private nodeUpdated(event: any) {
    if (event.name === 'Number of Inputs') {
      if (event.option) {
        if (event.option.value > this.numberOfInputs) {
          this.addInputs(event.option.value - this.numberOfInputs);
        } else if (event.option.value < this.numberOfInputs) {
          this.removeInputs(this.numberOfInputs - event.option.value);
        }
      }
    }

    // TODO
    /*
    if (event.name === 'Number of Outputs') {
      if (event.option) {
        if (event.option.value > this.numberOfOutputs) {
          this.addOutputInterface(event.option.value - this.numberOfOutputs);
        } else if (event.option.value < this.numberOfOutputs) {
          this.addOutputInterface(this.numberOfOutputs - event.option.value);
        }
      }
    }
    */
  }

  private addInputs(n: number) {
    Array.from(Array(n).keys()).forEach((i: number) => {
      this.addInputInterface(`Input ${this.numberOfInputs + i + 1}`);
    });
    this.numberOfInputs += n;
  }

  private removeInputs(n: number) {
    Array.from(Array(n).keys()).forEach((i: number) => {
      this.removeInterface(`Input ${this.numberOfInputs - i}`);
    });
    this.numberOfInputs -= n;
  }
}
