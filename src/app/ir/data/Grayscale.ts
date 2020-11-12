class Grayscale {
  constructor(
  ) {
  }

  static build(options: Map<string, any>): Grayscale {
    return new Grayscale();
  }

  public initCode(): string {
    return 'transforms.Grayscale()';
  }
}

export default Grayscale;
