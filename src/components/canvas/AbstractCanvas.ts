import { OptionPlugin } from '@baklavajs/plugin-options-vue';
import { ViewPlugin } from '@baklavajs/plugin-renderer-vue';
import Vector from '@/baklava/options/Vector.vue';
import Integer from '@/baklava/options/Integer.vue';
import Dropdown from '@/baklava/options/Dropdown.vue';
import CustomNode from '@/baklava/CustomNode.vue';

export default abstract class AbstractCanvas {
  protected option: OptionPlugin = new OptionPlugin();
  protected view: ViewPlugin = new ViewPlugin();

  constructor() {
    this.view.registerOption('VectorOption', Vector);
    this.view.registerOption('IntegerOption', Integer);
    this.view.registerOption('DropdownOption', Dropdown);

    this.view.components.node = CustomNode;
  }

  public get optionPlugin(): OptionPlugin {
    return this.option;
  }

  public get viewPlugin(): ViewPlugin {
    return this.view;
  }
}
