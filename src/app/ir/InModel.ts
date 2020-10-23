class InModel {
  constructor(
    public readonly dimension: bigint[],
  ) {}
  // eslint-disable-next-line class-methods-use-this
  public initCode(): string {
    return 'no init';
  }
}

export default InModel;
