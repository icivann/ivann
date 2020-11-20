import AbstractCanvas from '@/components/canvas/AbstractCanvas';
import { DataCategories, DataNodes } from '@/nodes/data/Types';

import InData from '@/nodes/data/InData';
import OutData from '@/nodes/data/OutData';
import ToTensor from '@/nodes/data/ToTensor';
import Grayscale from '@/nodes/data/Grayscale';
import LoadCsv from '@/nodes/data/LoadCsv';
import LoadImages from '@/nodes/data/LoadImages';
import LoadCustom from '@/nodes/data/LoadCustom';

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
      category: DataCategories.Loading,
      nodes: [
        {
          name: DataNodes.LoadCsv,
          node: LoadCsv,
        },
        {
          name: DataNodes.LoadImages,
          node: LoadImages,
        },
        {
          name: DataNodes.LoadCustom,
          node: LoadCustom,
        },
      ],
    },
  ];
}
