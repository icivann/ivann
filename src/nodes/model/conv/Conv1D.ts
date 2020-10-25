import Conv from '@/nodes/model/conv/Conv';
import { Layers, Nodes } from '@/nodes/model/Types';

export default class Conv1D extends Conv {
  type = Layers.Conv;
  name = Nodes.Conv1D;

  protected addKernelStride(): void {
    // TODO: Keras+Pytorch allow shortcut for specifying single int for all dimensions
    this.addOption('Kernel Size', 'IntOption', 1, undefined, {
      min: 1,
    });
    this.addOption('Stride', 'IntOption', 1, undefined, {
      min: 1,
    });
  }
}
