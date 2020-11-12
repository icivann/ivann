class ToTensor {
  constructor(
  ) {
  }

  static build(options: Map<string, any>): ToTensor {
    return new ToTensor();
  }

  public initCode(): string {
    return 'transforms.ToTensor()';
  }
}

export default ToTensor;
