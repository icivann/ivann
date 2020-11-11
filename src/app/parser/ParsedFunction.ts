class ParsedFunction {
  constructor(
    public readonly name: string,
    public readonly body: string,
    public readonly args: string[],
  ) {
  }

  public signature(): string {
    return `def ${this.name}(${this.args.join(', ')})`;
  }

  public toString(): string {
    return `${this.signature()}:${this.body}`;
  }
}

export default ParsedFunction;
