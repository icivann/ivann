import { Node } from '@baklavajs/core';
import { Layers, Nodes } from '@/nodes/model/Types';

export default class Custom extends Node {
  type = Layers.Custom;
  name = Nodes.Custom;

  private numberOfInputs = 0;

  constructor() {
    super();
    // TODO FE-38 Add minimum value of 0 to this option.
    this.addOption('Number of Inputs', 'IntOption', 0);
    this.addOutputInterface('Output');
    this.events.update.addListener(this, (event: any) => {
      this.nodeUpdated(event);
    });
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
