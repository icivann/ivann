import Conv from '@/nodes/model/conv/Conv';
import { Layers, Nodes } from '@/nodes/model/Types';

export default class Conv3D extends Conv {
  type = Layers.Conv;
  name = Nodes.Conv3D;

  protected addKernelStride(): void {
    // TODO: Keras+Pytorch allow shortcut for specifying single int for all dimensions
    this.addOption('Kernel Size', 'VectorOption', [1, 1, 1], undefined, {
      min: [1, 1, 1],
    });
    this.addOption('Stride', 'VectorOption', [1, 1, 1], undefined, {
      min: [1, 1, 1],
    });
  }
}
