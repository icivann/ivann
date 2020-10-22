import { OptionPlugin } from '@baklavajs/plugin-options-vue';
import { ViewPlugin } from '@baklavajs/plugin-renderer-vue';
import Vector from '@/baklava/options/Vector.vue';
import Integer from '@/baklava/options/Integer.vue';
import Dropdown from '@/baklava/options/Dropdown.vue';
import { Editor } from '@baklavajs/core';

export default abstract class AbstractCanvas {
  protected option = new OptionPlugin();
  protected view = new ViewPlugin();

  public get optionPlugin(): OptionPlugin {
    return this.option;
  }

  public get viewPlugin(): ViewPlugin {
    return this.view;
  }

  public registerOptions(): void {
    this.view.registerOption('VectorOption', Vector);
    this.view.registerOption('IntegerOption', Integer);
    this.view.registerOption('DropdownOption', Dropdown);
  }

  public abstract registerNodes(editor: Editor): void;
}
