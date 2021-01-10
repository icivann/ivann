import AbstractCanvas from '@/components/canvas/AbstractCanvas';
import { DataCategories, DataNodes } from '@/nodes/data/Types';

import OutData from '@/nodes/data/OutData';
import ToTensor from '@/nodes/data/ToTensor';
import Grayscale from '@/nodes/data/Grayscale';
import LoadCsv from '@/nodes/data/LoadCsv';
import LoadImages from '@/nodes/data/LoadImages';
import { Editor } from '@baklavajs/core';
import DataCustom from '@/nodes/data/DataCustom';

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
      ],
    },
  ];

  customNodeType = DataCustom;
  customNodeName = DataNodes.DataCustom;

  public registerNodes(editor: Editor) {
    super.registerNodes(editor);
    editor.registerNodeType(DataNodes.OutData, OutData);
  }
}
