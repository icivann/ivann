import AbstractNodesTab from '@/components/tabs/nodes/AbstractNodesTab';
import { SearchItem } from '@/components/SearchUtils';
import { ParsedFile } from '@/store/codeVault/types';
import { CommonNodes } from '@/nodes/common/Types';

export default class CustomNodesTab extends AbstractNodesTab {
  private items: SearchItem[] = [];

  public constructor(files: ParsedFile[]) {
    super();
    this.updateFiles(files);
  }

  public get searchItems(): SearchItem[] {
    return this.items;
  }

  public updateFiles(files: ParsedFile[]): void {
    const mapped: SearchItem[] = files.map((file) => ({
      category: file.filename,
      nodes: file.functions.map((func) => ({
        name: CommonNodes.Custom,
        displayName: func.name,
        options: func,
      })),
    }));

    // TODO: Find a way to use searchItems getter instead of array reference
    this.items.splice(0, this.items.length, ...mapped);
  }
}
