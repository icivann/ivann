import AbstractCanvas from '@/components/canvas/AbstractCanvas';
import { DataCategories, DataNodes } from '@/nodes/data/Types';

import InData from '@/nodes/data/InData';
import OutData from '@/nodes/data/OutData';
import ToTensor from '@/nodes/data/ToTensor';
import Grayscale from '@/nodes/data/Grayscale';
import { Editor } from '@baklavajs/core';

export default class DataCanvas extends AbstractCanvas {
  public nodeList = [
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
  ];

  public registerNodes(editor: Editor) {
    super.registerNodes(editor);
    editor.registerNodeType(DataNodes.InData, InData);
    editor.registerNodeType(DataNodes.OutData, OutData);
  }
}
