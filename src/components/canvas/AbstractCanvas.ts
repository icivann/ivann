import { OptionPlugin } from '@baklavajs/plugin-options-vue';
import { Editor } from '@baklavajs/core';
import { NodeConstructor } from '@baklavajs/core/dist/baklavajs-core/types/index.d';
import { CommonNodes } from '@/nodes/common/Types';
import Custom from '@/nodes/common/Custom';

export default abstract class AbstractCanvas {
  public abstract nodeList: {
    category: string;
    nodes: {
      name: string;
      node: NodeConstructor;
    }[];
  }[];

  protected option: OptionPlugin = new OptionPlugin();

  public get optionPlugin(): OptionPlugin {
    return this.option;
  }

  public registerNodes(editor: Editor) {
    this.nodeList.forEach(({ category, nodes }) => {
      nodes.forEach(({ name, node }) => {
        editor.registerNodeType(name, node, category);
      });
    });
    editor.registerNodeType(CommonNodes.Custom, Custom);
  }
}
