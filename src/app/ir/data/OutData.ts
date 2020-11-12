class OutData {
  constructor(
    batchSize: BigInt,
  ) {
  }

  static build(options: Map<string, any>): OutData {
    return new OutData(options.get('BatchSize'));
  }
}

export default OutData;
