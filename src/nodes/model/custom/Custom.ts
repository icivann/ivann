import { Node } from '@baklavajs/core';
import { Layers, Nodes } from '@/nodes/model/Types';
import parse from '@/app/parser/parser';
import OptionParams from '@/baklava/OptionParams';

export enum CustomOptions{
  InlineCode = 'Inline Code',
}
export default class Custom extends Node {
  type = Layers.Custom;
  name = Nodes.Custom;

  private inputNames: string[] = [];

  private inlineCodeParams = new OptionParams(false);

  constructor() {
    super();
    this.addOption(CustomOptions.InlineCode, 'TextAreaOption', '', undefined, { params: this.inlineCodeParams });
    this.addOutputInterface('Output');
    this.events.update.addListener(this, (event: any) => {
      this.nodeUpdated(event);
    });
  }

  private nodeUpdated(event: any) {
    if (event.name === CustomOptions.InlineCode) {
      const code = event.option.value;

      if (code === '') {
        this.removeAllInputs();
      } else {
        parse(code)
          .then((functions) => {
            this.inlineCodeParams.hasError = false;
            console.log(functions);
            if (functions.length > 0) {
              const func = functions[0];
              this.setInputs(func.args);
            }
          })
          .catch((err: Error) => {
            this.inlineCodeParams.hasError = true;
            // TODO Do feedback
            throw err;
          });
      }
    }
  }

  private setInputs(inputNames: string[]) {
    this.removeAllInputs();
    inputNames.forEach((inputName: string) => {
      this.addInputInterface(inputName);
    });
    this.inputNames = inputNames;
  }

  private removeAllInputs() {
    this.inputNames.forEach((inputName: string) => {
      this.removeInterface(inputName);
    });
    this.inputNames = [];
  }
}
