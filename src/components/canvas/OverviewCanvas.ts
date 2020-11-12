import AbstractCanvas from '@/components/canvas/AbstractCanvas';
import Model from '@/nodes/overview/Model';
import { OverviewCategories, OverviewNodes } from '@/nodes/overview/Types';
import { CommonNodes } from '@/nodes/common/Types';
import Custom from '@/nodes/common/Custom';

export default class OverviewCanvas extends AbstractCanvas {
  nodeList = [
    {
      category: OverviewCategories.Model,
      nodes: [
        {
          name: OverviewNodes.ModelNode,
          node: Model,
        },
      ],
    },
    {
      category: OverviewCategories.Custom,
      nodes: [
        {
          name: CommonNodes.Custom,
          node: Custom,
        },
      ],
    },
  ];
}
