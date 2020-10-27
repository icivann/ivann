import { OptionPlugin } from '@baklavajs/plugin-options-vue';
import { Editor } from '@baklavajs/core';

export default abstract class AbstractCanvas {
  protected option: OptionPlugin = new OptionPlugin();

  public get optionPlugin(): OptionPlugin {
    return this.option;
  }

  public abstract registerNodes(editor: Editor): void;
}
