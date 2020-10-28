export default class ToDevice {
  constructor(
    public device: string,
  ) {}

  static build(options: Map<string, any>): ToDevice {
    return new ToDevice('GPU_0');
  }
}
