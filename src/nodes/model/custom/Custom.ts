import { Node } from '@baklavajs/core';
import { Layers, Nodes } from '@/nodes/model/Types';
import parse from '@/app/parser/parser';

export enum CustomOptions{
  InlineCode = 'Inline Code',
}
export default class Custom extends Node {
  type = Layers.Custom;
  name = Nodes.Custom;

  private inputNames: string[] = [];

  private code = '';

  constructor() {
    super();
    this.addOption(CustomOptions.InlineCode, 'TextAreaOption',
      { text: this.code, hasError: false });
    this.addOutputInterface('Output');
    this.events.update.addListener(this, (event: any) => {
      this.nodeUpdated(event);
    });
  }

  private nodeUpdated(event: any) {
    if (event.name === CustomOptions.InlineCode) {
      const code = event.option.value.text;

      /* If the code did not change (probably the hasError did), do not parse it */
      if (code === this.code) {
        return;
      }

      this.code = code;

      if (code === '') {
        this.removeAllInputs();
      } else {
        parse(code)
          .then((functions) => {
            this.setError(false);
            console.log(functions);
            if (functions.length > 0) {
              const func = functions[0];
              this.setInputs(func.args);
            }
          })
          .catch((err: Error) => {
            this.setError(true);
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

  private setError(error: boolean) {
    this.setOptionValue(
      CustomOptions.InlineCode,
      { text: this.getOptionValue(CustomOptions.InlineCode).text, hasError: error },
    );
  }
}
