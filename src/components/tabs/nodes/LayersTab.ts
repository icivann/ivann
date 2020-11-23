import AbstractNodesTab from '@/components/tabs/nodes/AbstractNodesTab';
import { convertToSearch, modify, SearchItem } from '@/components/SearchUtils';
import EditorManager from '@/EditorManager';
import { ModelCategories, ModelNodes } from '@/nodes/model/Types';

export default class LayersTab extends AbstractNodesTab {
  private readonly items: SearchItem[];

  public constructor(names: Set<string>) {
    super();
    this.items = convertToSearch(EditorManager.getInstance().modelCanvas.nodeList);
    this.items = modify(this.items, ModelCategories.IO, ModelNodes.InModel, {
      name: ModelNodes.InModel,
      displayName: 'Input',
      names,
    });
    this.items = modify(this.items, ModelCategories.IO, ModelNodes.OutModel, {
      name: ModelNodes.OutModel,
      displayName: 'Output',
      names,
    });
  }

  public get searchItems(): SearchItem[] {
    return this.items;
  }
}
