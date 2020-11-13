import AbstractCanvas from '@/components/canvas/AbstractCanvas';
import { CommonNodes } from '@/nodes/common/Types';
import Custom from '@/nodes/common/Custom';

import { DataCategories, DataNodes } from '@/nodes/data/Types';

import InData from '@/nodes/data/InData';
import OutData from '@/nodes/data/OutData';
import ToTensor from '@/nodes/data/ToTensor';
import Grayscale from '@/nodes/data/Grayscale';

export default class DataCanvas extends AbstractCanvas {
  public nodeList = [
    {
      category: DataCategories.IO,
      nodes: [
        {
          name: DataNodes.InData,
          node: InData,
        },
        {
          name: DataNodes.OutData,
          node: OutData,
        },
      ],
    },
    {
      category: DataCategories.Transform,
      nodes: [
        {
          name: DataNodes.ToTensor,
          node: ToTensor,
        },
        {
          name: DataNodes.Grayscale,
          node: Grayscale,
        },
      ],
    },
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
