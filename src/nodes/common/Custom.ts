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
  name: string = CommonNodes.Custom;

  private inputNames: string[] = [];

  constructor(parsedFunction?: ParsedFunction) {
    super();
    if (parsedFunction) {
      this.state.parsedFunction = parsedFunction;
      this.updateNode();
    }
  }

  public load(state: INodeState) {
    this.updateNode(state);
    super.load(state);
  }

  public setParsedFunction(parsedFunction?: ParsedFunction) {
    this.state.parsedFunction = parsedFunction;
  }

  public addInput(name: string): void {
    this.addInputInterface(name);
  }

  public remInteface(name: string): void {
    this.removeInterface(name);
  }

  public getParsedFunction(savedState?: INodeState): (ParsedFunction | undefined) {
    const { parsedFunction } = savedState ? savedState.state : this.state;
    if (parsedFunction) {
      // Conversion is necessary because Baklava State saves as generic `any`.
      return new ParsedFunction(
        parsedFunction.name,
        parsedFunction.body,
        parsedFunction.args,
        parsedFunction.filename,
      );
    }
    return undefined;
  }

  private updateNode(savedState?: INodeState) {
    const parsedFunction = this.getParsedFunction(savedState);
    if (!parsedFunction) {
      this.name = CommonNodes.Custom;
      this.removeAllInputs();
      this.removeOutput();
    } else {
      this.name = parsedFunction.name;
      this.setInputs(parsedFunction.args);
      if (parsedFunction.containsReturn()) {
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

  public addOutput() {
    this.addOutputInterface('Output');
  }

  public removeOutput() {
    if (this.interfaces.has('Output')) {
      this.removeInterface('Output');
    }
  }
}
