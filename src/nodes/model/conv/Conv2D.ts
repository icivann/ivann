import Conv, { ConvOptions } from '@/nodes/model/conv/Conv';
import { Nodes } from '@/nodes/model/Types';

export default class Conv2D extends Conv {
  type = Nodes.Conv2D;
  name = Nodes.Conv2D;

  protected addKernelStride(): void {
    // TODO: Keras+Pytorch allow shortcut for specifying single int for all dimensions
    this.addOption(ConvOptions.KernelSize, 'VectorOption', [1, 1], undefined, {
      min: [1, 1],
    });
    this.addOption(ConvOptions.Stride, 'VectorOption', [1, 1], undefined, {
      min: [1, 1],
    });
  }
}
