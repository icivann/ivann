class ParsedFunction {
  constructor(
    public readonly name: string,
    public readonly body: string,
    public readonly args: string[],
  ) {
  }

  public toString(): string {
    return `def ${this.name}(${this.args.join(', ')}):\n`
      + `${this.body}`;
  }
}

export default ParsedFunction;
