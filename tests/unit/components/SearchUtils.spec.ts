import { NodeListItem } from '@/components/canvas/AbstractCanvas';
import { convertToSearch, SearchItem } from '@/components/SearchUtils';
import { Node } from '@baklavajs/core';

class MockNode extends Node {
  name = 'Mock';
  type = 'Mock';
}

describe('Init node list', () => {
  it('Init empty list', () => {
    const list: NodeListItem[] = [];
    return expect(convertToSearch(list)).toEqual([]);
  });

  it('Init empty category', () => {
    const categoryName = 'MyCategory';
    const list: NodeListItem[] = [{
      category: categoryName,
      nodes: [],
    }];

    const expected: SearchItem[] = [{
      category: categoryName,
      nodes: [],
    }];
    return expect(convertToSearch(list)).toStrictEqual(expected);
  });

  it('Init populated category', () => {
    const categoryName = 'MyCategory';
    const list: NodeListItem[] = [{
      category: categoryName,
      nodes: [{
        name: MockNode.name,
        node: MockNode,
      }],
    }];

    const expected: SearchItem[] = [{
      category: categoryName,
      nodes: [{
        name: MockNode.name,
        displayName: MockNode.name,
      }],
    }];
    return expect(convertToSearch(list)).toStrictEqual(expected);
  });
});
