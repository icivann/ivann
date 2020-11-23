import AbstractNodesTab from '@/components/tabs/nodes/AbstractNodesTab';
import { convertToSearch, modify, SearchItem } from '@/components/SearchUtils';
import EditorManager from '@/EditorManager';
import { ModelCategories, ModelNodes } from '@/nodes/model/Types';

export default class LayersTab extends AbstractNodesTab {
  public readonly searchItems: SearchItem[];

  public constructor(names: Set<string>) {
    super();
    this.searchItems = convertToSearch(EditorManager.getInstance().modelCanvas.nodeList);
    this.searchItems = modify(this.searchItems, ModelCategories.IO, ModelNodes.InModel, {
      name: ModelNodes.InModel,
      displayName: 'Input',
      names,
    });
    this.searchItems = modify(this.searchItems, ModelCategories.IO, ModelNodes.OutModel, {
      name: ModelNodes.OutModel,
      displayName: 'Output',
      names,
    });
  }
}
