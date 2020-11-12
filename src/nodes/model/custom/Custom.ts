import { Node } from '@baklavajs/core';
import { Nodes } from '@/nodes/model/Types';
import { INodeState } from '@baklavajs/core/dist/baklavajs-core/types/state.d';
import ParsedFunction from '@/app/parser/ParsedFunction';

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
    this.addOption('Select Function', 'CodeVaultButtonOption', undefined, undefined, { customNode: this });
  }

  public load(state: INodeState) {
    super.load(state);
    this.updateNode();
  }

  public setInlineCode(inlineCode?: ParsedFunction) {
    this.state.inlineCode = inlineCode;
    this.updateNode();
  }

  public getInlineCode(): (ParsedFunction | undefined) {
    const { inlineCode } = this.state;
    if (this.state.inlineCode) {
      return new ParsedFunction(inlineCode.name, inlineCode.body, inlineCode.args);
    }
    return undefined;
  }

  private updateNode() {
    const inlineCode = this.getInlineCode();
    if (!inlineCode) {
      this.name = Nodes.Custom;
      this.removeAllInputs();
      this.removeOutput();
    } else {
      this.name = inlineCode.name;
      this.setInputs(inlineCode.args);
      this.addOutput();
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
