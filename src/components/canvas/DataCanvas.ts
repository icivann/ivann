import AbstractCanvas from '@/components/canvas/AbstractCanvas';
import { DataCategories, DataNodes } from '@/nodes/data/Types';

import OutData from '@/nodes/data/OutData';
import ToTensor from '@/nodes/data/ToTensor';
import Grayscale from '@/nodes/data/Grayscale';
import LoadCsv from '@/nodes/data/LoadCsv';
import LoadImages from '@/nodes/data/LoadImages';
import { Editor } from '@baklavajs/core';

export default class DataCanvas extends AbstractCanvas {
  public nodeList = [
    {
      category: DataCategories.Transform,
      nodes: [
        {
          name: DataNodes.ToTensor,
          node: ToTensor,
          img: 'transform-icon.svg',
        },
        {
          name: DataNodes.Grayscale,
          node: Grayscale,
          img: 'transform-icon.svg',
        },
      ],
    },
    {
      category: DataCategories.Loading,
      nodes: [
        {
          name: DataNodes.LoadCsv,
          node: LoadCsv,
          img: 'input-icon.svg',
        },
        {
          name: DataNodes.LoadImages,
          node: LoadImages,
          img: 'input-icon.svg',
        },
      ],
    },
  ];

  public registerNodes(editor: Editor) {
    super.registerNodes(editor);
    editor.registerNodeType(DataNodes.OutData, OutData);
  }
}
