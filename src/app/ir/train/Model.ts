export default class Model {
  constructor(
    public nInputs: BigInt,
    public nOutputs: BigInt,
  ) {}

  static build(options: Map<string, any>): Model {
    return new Model(BigInt(1), BigInt(1));
  }
}
