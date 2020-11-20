import { nodeName } from '@/app/ir/irCommon';

class LoadImages {
  constructor(
    public readonly name: string,
  ) {
  }

  static build(options: Map<string, any>): LoadImages {
    return new LoadImages(
      options.get(nodeName),
    );
  }

  public initCode(name: string): string[] {
    const path = `${name}_path`;
    return [
      `self.${name} = [os.path.join(${path}, file) for file in listdir(${path}) if os.path.isfile(os.path.join(${path}, file))]`,
    ];
  }

  public callCode(name: string): string[] {
    return [
      `${name} = Image(${name})`,
    ];
  }
}

export default LoadImages;
