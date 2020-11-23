import AbstractNodesTab from '@/components/tabs/nodes/AbstractNodesTab';
import { convertToSearch, modify, SearchItem } from '@/components/SearchUtils';
import EditorManager from '@/EditorManager';
import { DataCategories, DataNodes } from '@/nodes/data/Types';

export default class DataComponentsTab extends AbstractNodesTab {
  public readonly searchItems: SearchItem[];

  public constructor(names: Set<string>) {
    super();
    this.searchItems = convertToSearch(EditorManager.getInstance().dataCanvas.nodeList);
    this.searchItems = modify(this.searchItems, DataCategories.IO, DataNodes.InData, {
      name: DataNodes.InData,
      displayName: 'Input',
      names,
    });
    this.searchItems = modify(this.searchItems, DataCategories.IO, DataNodes.OutData, {
      name: DataNodes.OutData,
      displayName: 'Output',
      names,
    });
  }
}
