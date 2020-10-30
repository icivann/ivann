import { ReLUOptions } from '@/nodes/pytorch model/ReLUBaklava';

export default class ReLU {
constructor(
  public readonly inplace: boolean,
) {
}
 
static build(options: Map<string, any>): ReLU {
  return new ReLU(
    options.get(ReLUOptions.Inplace),
  );
  
  }
  
  public initCode(): string{
    return `ReLU(inplace=${this.inplace})`;
  }
}
  