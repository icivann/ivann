import re
import os

type_map = {
  "int" : "IntOption",
  "Union[T,Tuple[T]]":"VectorOption",
  "str":"DropdownOption",
  "bool":"TickBoxOption"
}

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
  
  parameter_map = parse_options(parameter_name_and_types)

  return class_name, parameter_map

def create_baklava(class_name, option_map: dict):
  f_path = os.path.join(f"{class_name.capitalize()}.ts")
  baklava_file = open(f_path, "w")
  
  option_enums_header = f"export enum {class_name.capitalize()}Options {{"
  option_enums_values = []
  options_add = []
  for k,v in option_map.items():
    option_name = k.capitalize()
    enumValue = k.replace("_"," ").capitalize()
    option_enums_values.append(f"{option_name} = '{enumValue}'") 

    option_type, default_value = v
    option_type = type_map[option_type]
    default_value = default_value if default_value else '//TODO: Add default value'

    option_add = f"""this.addOption({class_name}Options.{option_name}, '{option_type}', {default_value});"""
    options_add.append(option_add)

  options_add = "\n  ".join(options_add)
  option_enums_values = ",\n  ".join(option_enums_values)
  option_enums = f"{option_enums_header}\n  {option_enums_values}\n}}"
  # option_enums = "\n".join(option_enums)
  

  print("options enums\n\n", option_enums)

  contents =  f"""import {{ Node }} from '@baklavajs/core';
import {{ Layers, Nodes }} from '@/nodes/model/Types';
{option_enums}
export default class {class_name} extends Node {{
constructor() {{
super();
  this.addInputInterface('Input');
  this.addOutputInterface('Output');
  {options_add}
  }}
}}
  """

  baklava_file.write(contents)

def create_ir_node(class_name, option_map: dict):
  f_path = os.path.join(f"{class_name.lower()}.ts")
  ir_file = open(f_path, "w")
  
  option_enums_header = f"export enum {class_name.capitalize()}Options {{"
  option_enums_values = []
  options_add = []
  fields = []
  for k,v in option_map.items():
    option_name = k.capitalize()
    enumValue = k.replace("_"," ").capitalize()
    option_enums_values.append(f"{option_name} = '{enumValue}'") 

    option_type, default_value = v
    option_type = type_map[option_type]
    default_value = default_value if default_value else '//TODO: Add default value'

    option_add = f"""this.addOption({class_name}Options.{option_name}, '{option_type}', {default_value});"""
    options_add.append(option_add)

  options_add = "\n  ".join(options_add)
  option_enums_values = ",\n  ".join(option_enums_values)
  option_enums = f"{option_enums_header}\n  {option_enums_values}\n}}"
  # option_enums = "\n".join(option_enums)
  

  print("options enums\n\n", option_enums)

  contents =  f"""import {{ {class_name}Options }} from 'TODO';
import {{ ModelNode }} from 'TODO';

export default class {class_name} {{
constructor(
  {fields}
  
}}
  """

  ir_file.write(contents)

if __name__ == "__main__":
  string = "torch.nn.Conv1d(in_channels: int, out_channels: int, kernel_size: Union[T, Tuple[T]], stride: Union[T, Tuple[T]] = 1, padding: Union[T, Tuple[T]] = 0, dilation: Union[T, Tuple[T]] = 1, groups: int = 1, bias: bool = True, padding_mode: str = 'zeros')"
  class_name, options_map = parse(string)

  create_baklava(class_name, options_map)
  create_ir_node(class_name, options_map)