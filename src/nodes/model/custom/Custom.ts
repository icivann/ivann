import { Node } from '@baklavajs/core';
import { Nodes } from '@/nodes/model/Types';
import { INodeState } from '@baklavajs/core/dist/baklavajs-core/types/state.d';
import ParsedFunction from '@/app/parser/ParsedFunction';

// TODO CORE-58 Change InlineCode Option to use state.parsedFunction
export enum CustomOptions {
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

  public setParsedFunction(parsedFunction?: ParsedFunction) {
    this.state.parsedFunction = parsedFunction;
    this.updateNode();
  }

  public getParsedFunction(): (ParsedFunction | undefined) {
    const { parsedFunction } = this.state;
    if (this.state.parsedFunction) {
      // Conversion is necessary because Baklava State saves as generic `any`.
      return new ParsedFunction(parsedFunction.name, parsedFunction.body, parsedFunction.args);
    }
    return undefined;
  }

  public setParsedFileName(parsedFileName?: string) {
    this.state.parsedFileName = parsedFileName;
  }

  public getParsedFileName(): (string | undefined) {
    return this.state.parsedFileName;
  }

  private updateNode() {
    const parsedFunction = this.getParsedFunction();
    if (!parsedFunction) {
      this.name = Nodes.Custom;
      this.removeAllInputs();
      this.removeOutput();
    } else {
      this.name = parsedFunction.name;
      this.setInputs(parsedFunction.args);
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
