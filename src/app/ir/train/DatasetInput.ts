export default class DatasetInput {
  constructor(
    public dataset: string,
  ) {}

  static build(options: Map<string, any>): DatasetInput {
    return new DatasetInput('MNIST');
  }
}
