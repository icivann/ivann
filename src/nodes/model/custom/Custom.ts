import { Node } from '@baklavajs/core';
import { Nodes } from '@/nodes/model/Types';
import parse from '@/app/parser/parser';
import { INodeState } from '@baklavajs/core/dist/baklavajs-core/types/state.d';

// TODO CORE-58 Change this to use state.inlineCode
export enum CustomOptions{
  InlineCode = 'Inline Code',
}
export default class Custom extends Node {
  type = Nodes.Custom;
  name: string = Nodes.Custom;

  private inputNames: string[] = [];

  constructor() {
    super();
    this.addOption('Enter Func', 'CodeVaultButtonOption', undefined, undefined, { customNode: this });
  }

  public load(state: INodeState) {
    super.load(state);
    this.updateNode();
  }

  public setInlineCode(inlineCode: string) {
    this.state.inlineCode = inlineCode;
    this.updateNode();
  }

  public getInlineCode() {
    return this.state.inlineCode ? this.state.inlineCode : '';
  }

  private updateNode() {
    if (this.getInlineCode() === '') {
      this.name = Nodes.Custom;
      this.removeAllInputs();
      this.removeOutput();
    } else {
      const functions = parse(this.getInlineCode());
      if (!(functions instanceof Error) && functions.length > 0) {
        const func = functions[0];
        this.name = func.name;
        this.setInputs(func.args);
        this.addOutput();
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

  private addOutput() {
    this.addOutputInterface('Output');
  }

  private removeOutput() {
    if (this.interfaces.has('Output')) {
      this.removeInterface('Output');
    }
  }
}
