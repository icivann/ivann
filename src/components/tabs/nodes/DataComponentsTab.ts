import AbstractNodesTab from '@/components/tabs/nodes/AbstractNodesTab';
import { convertToSearch, modify, SearchItem } from '@/components/SearchUtils';
import EditorManager from '@/EditorManager';
import { DataCategories, DataNodes } from '@/nodes/data/Types';

export default class DataComponentsTab extends AbstractNodesTab {
  private readonly items: SearchItem[];

  public constructor(names: Set<string>) {
    super();
    this.items = convertToSearch(EditorManager.getInstance().dataCanvas.nodeList);
    this.items = modify(this.items, DataCategories.IO, DataNodes.InData, {
      name: DataNodes.InData,
      displayName: 'Input',
      names,
    });
    this.items = modify(this.items, DataCategories.IO, DataNodes.OutData, {
      name: DataNodes.OutData,
      displayName: 'Output',
      names,
    });
  }

  public get searchItems(): SearchItem[] {
    return this.items;
  }
}
