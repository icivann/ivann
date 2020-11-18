import { Node } from '@baklavajs/core';
import { INodeState } from '@baklavajs/core/dist/baklavajs-core/types/state.d';
import ParsedFunction from '@/app/parser/ParsedFunction';
import { CommonNodes } from '@/nodes/common/Types';

// TODO CORE-58 Change InlineCode Option to use state.parsedFunction
export enum CustomOptions {
  Code = 'Code'
}
export default class Custom extends Node {
  type = CommonNodes.Custom;
  name: string = CommonNodes.Custom; // de

  private inputNames: string[] = [];

  constructor(parsedFunction?: ParsedFunction) {
    super();
    if (parsedFunction) {
      this.state.parsedFunction = parsedFunction;
      this.updateNode();
    }
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
      this.name = CommonNodes.Custom;
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
