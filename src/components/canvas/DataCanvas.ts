import AbstractCanvas from '@/components/canvas/AbstractCanvas';
import { DataCategories } from '@/nodes/data/Types';
import { CommonNodes } from '@/nodes/common/Types';
import Custom from '@/nodes/common/Custom';

export default class DataCanvas extends AbstractCanvas {
  public nodeList = [
    {
      category: DataCategories.Custom,
      nodes: [
        {
          name: CommonNodes.Custom,
          node: Custom,
        },
      ],
    },
  ];
}
