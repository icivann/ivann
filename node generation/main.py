import re
import os

def parse_options(string):
  ret = {}
  split = string.split(":")

  name = split[0]
  default_value = None
  split = split[1:]

  for i, x in enumerate(split[:-1]):
    x = x.replace(" ", "")
    x = x[::-1]
    x_split = x.split(",", 1)
    x_split = list(map(lambda y: y[::-1], x_split))
    rest = x_split[1].split("=")
    default_value = rest[1] if len(rest) > 1 else None
    x_type = rest[0]

    # print(name, x_type, default_value)
    ret[name] = (x_type, default_value)
    name = x_split[0]

  x_split = split[-1]
  x_split = x_split.replace(" ", "")
  rest = x_split.split("=")
  default_value = rest[1] if len(rest) > 1 else None
  x_type = rest[0]

  # print(name, x_type, default_value)
  ret[name] = (x_type, default_value)

  return ret

def parse(x):
  x = x[9:]

  split = x.split('(', 1)
  class_name = split[0]
  parameter_name_and_types = split[1][:-1]
  parameter_name_and_types.replace('[T, Tuple[T]]', 'Union')
  
  f''


  for x in parameter_name_and_types:
    print(x)
  print()
  quit()
  
  options = {}

  print(class_name)
  print(parameter_name_and_types)

  print("options \n")

  for option in parameter_name_and_types:
    option = option.replace(" ", "")
    split_option = option.split(":")
    option_name = split_option[0]
    print(split_option)
    rest = split_option[1].split('=')
    default_value = rest[1] if len(rest) > 1 else None
    option_type = rest[0]
    
    print(option_name, option_type, default_value)
    
    options[option_name] = option_type


def create_baklava(class_name, option_map: dict):
  f_path = os.path.join("Conv1D.ts")
  f = open(f_path, "w")
  ts = "import { Node } from '@baklavajs/core'; \nimport { Layers, Nodes } from '@/nodes/model/Types';"
  
  option_enums = [f"export enum {class_name.capitalize()}{{"]
  for k,v in option_map.items():
    x = f"{enumValue}= ' {k.capitalize()}"
    enumValue = k.capitalize().replace(" ","")
    option_enums.append(f"{enumValue.capitalize()} = ' {k.capitalize()}'") 

    options_add.append(f'this.addOption('{})

  option_enums.append("}")


  

  [ts,  
  option_enums,
  option_enums,
  'export default class MaxPool2D extends Node {',
  'constructor() {',
  'super();',
  '  this.addInputInterface(\'Input\');',
  '  this.addOutputInterface(\'Output\');',
  option_add
  ].join('\n')
  

if __name__ == "__main__":
  string = "torch.nn.Conv1d(in_channels: int, out_channels: int, kernel_size: Union[T, Tuple[T]], stride: Union[T, Tuple[T]] = 1, padding: Union[T, Tuple[T]] = 0, dilation: Union[T, Tuple[T]] = 1, groups: int = 1, bias: bool = True, padding_mode: str = 'zeros')"
  parse(string)