import { NodeListItem } from '@/components/canvas/AbstractCanvas';
import {
  convertToSearch, modify, SearchItem, SearchNode,
} from '@/components/SearchUtils';
import { Node } from '@baklavajs/core';

class MockNode extends Node {
  name = 'Mock';
  type = 'Mock';
}

describe('Init node list', () => {
  const categoryName = 'MyCategory';

  it('Init empty list', () => {
    const list: NodeListItem[] = [];
    return expect(convertToSearch(list)).toEqual([]);
  });

  it('Init empty category', () => {
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
    const list: NodeListItem[] = [{
      category: categoryName,
      nodes: [{
        name: MockNode.name,
        node: MockNode,
      }],
    }];

    const expected: SearchItem = {
      category: categoryName,
      nodes: [{
        name: MockNode.name,
        displayName: MockNode.name,
      }],
    };

    const searchItems = convertToSearch(list);
    expect(searchItems.length).toEqual(1);
    expect(searchItems[0].category).toEqual(expected.category);
    expect(searchItems[0].nodes.length).toEqual(expected.nodes.length);
    expect(searchItems[0].nodes[0].name).toEqual(expected.nodes[0].name);
    expect(searchItems[0].nodes[0].displayName).toEqual(expected.nodes[0].displayName);
  });
});

describe('Modify entries', () => {
  const category1 = 'Category1';
  const node1 = 'Node1';
  const node2 = 'Node2';
  const node3 = 'Node3';

  it('Modifes Display name on 1', () => {
    const searchItem: SearchItem = {
      category: category1,
      nodes: [
        {
          name: node1,
          displayName: node1,
        },
      ],
    };

    const expected: SearchItem = {
      category: category1,
      nodes: [
        {
          name: node1,
          displayName: node2,
        },
      ],
    };

    const replacement: SearchNode = { name: node1, displayName: node2 };

    expect(modify([searchItem], category1, node1, replacement))
      .toStrictEqual([expected]);
  });
});
