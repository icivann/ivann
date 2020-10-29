import re
import os

type_map = {
  "int": "IntOption",
  "Union": "VectorOption",
  "Optional[Union]": "VectorOption",
  "str": "DropdownOption",
  "bool": "TickBoxOption"
}

IR_map = {
  "int": "bigint",
  "Union": "[T]",
  "Optional[Union]": "[T]",
  "str": "enum",
  "bool": "boolean"
}

def parse_options(string):
  string = re.sub('(Union\[T,( )*Tuple\[([A-Z, .]*\]\]))', 'Union', string)

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


def create_baklava(class_name, option_map: dict, dimensions):
  f_path = os.path.join(f"{class_name.capitalize()}Baklava.ts")
  baklava_file = open(f_path, "w")

  option_enums_header = f"export enum {class_name.capitalize()}Options {{"
  option_enums_values = []
  options_add = []
  for k, v in option_map.items():
    option_name = k.capitalize()
    enumValue = k.replace("_", " ").capitalize()
    option_enums_values.append(f"{option_name} = '{enumValue}'")

    option_type, default_value = v
    option_type = type_map[option_type]
    default_value = default_value if default_value else '//TODO: Add default value'
    default_value = 'CheckboxValue.CHECKED' if default_value == 'True' else default_value

    option_add = f"""this.addOption({class_name}Options.{option_name}, TypeOptions.{option_type}, {default_value});"""
    options_add.append(option_add)

  options_add = "\n  ".join(options_add)
  option_enums_values = ",\n  ".join(option_enums_values)
  option_enums = f"{option_enums_header}\n  {option_enums_values}\n}}"
  # option_enums = "\n".join(option_enums)

  print("options enums\n\n", option_enums)

  contents = f"""import {{ Node }} from '@baklavajs/core';
import {{ Layers, Nodes }} from '@/nodes/model/Types';
import {{ TypeOptions }} from '@/nodes/model/BaklavaDisplayTypeOptions';

{option_enums}
export default class {class_name} extends Node {{
  type = Layers.;//TODO add layer type
  name = Nodes.{class_name.capitalize()};

constructor() {{
super();
  this.addInputInterface('Input');
  this.addOutputInterface('Output');
  {options_add}
  }}
}}
  """

  baklava_file.write(contents)


def create_ir_node(class_name, option_map: dict, dimensions):
  f_path = os.path.join(f"{class_name.lower()}.ts")
  ir_file = open(f_path, "w")

  fields = []
  build = []
  pythonCode =[]

  for k, v in option_map.items():
    field_name = k.lower()
    option_type, default_value = v
    option_type = IR_map[option_type]

    if option_type =="enum":
      option_type = field_name.capitalize()
      buildLine = f"get{field_name.capitalize()}(options.get({class_name}Options.{field_name}))"
    elif option_type == "bool":
      continue
    elif option_type == "[T]":

      buildLine = "["
      option_type = "["
      for i in range (dimensions[0]-1):
        buildLine+= f"  options.get({class_name}Options.{field_name}[{i}]), "
        option_type+= "bigint, "
      buildLine+= f"options.get({class_name}Options.{field_name})[{dimensions[0]-1}]], "
      option_type+= "bigint]"
    else:
      buildLine= f"options.get({class_name}Options.{field_name}),"


    fields.append(f"public readonly {field_name}: {option_type},")
    build.append(buildLine)
    pythonCode.append(f"{field_name}=${{this.{field_name}}}")

  fields = "\n  ".join(fields)
  build = "\n  ".join(build)
  pythonCode = ", ".join(pythonCode)

  contents = f"""import {{ {class_name}Options }} from '@/nodes/model/{class_name}';

export default class {class_name} {{
constructor(
  {fields}
) {{
}}
 
static build(options: Map<string, any>): {class_name} {{
  return new {class_name}(
    {build}
  );
  
  }}
  
  public initCode(): string{{
    return `{class_name}({pythonCode})`;
  }}
}}
  """

  ir_file.write(contents)


if __name__ == "__main__":
  string = "torch.nn.Conv1d(in_channels: int, out_channels: int, kernel_size: Union[T, Tuple[T]], stride: Union[T, Tuple[T]] = 1, padding: Union[T, Tuple[T]] = 0, dilation: Union[T, Tuple[T]] = 1, groups: int = 1, bias: bool = True, padding_mode: str = 'zeros')"
  string2 = "torch.nn.MaxPool2d(kernel_size: Union[T, Tuple[T, ...]], stride: Optional[Union[T, Tuple[T, ...]]] = None, padding: Union[T, Tuple[T, ...]] = 0, dilation: Union[T, Tuple[T, ...]] = 1, return_indices: bool = False, ceil_mode: bool = False)"
  string3 ="torch.nn.Conv2d(in_channels: int, out_channels: int, kernel_size: Union[T, Tuple[T, T]], stride: Union[T, Tuple[T, T]] = 1, padding: Union[T, Tuple[T, T]] = 0, dilation: Union[T, Tuple[T, T]] = 1, groups: int = 1, bias: bool = True, padding_mode: str = 'zeros')"
  string4 = "torch.nn.MaxPool2d(kernel_size: Union[T, Tuple[T, ...]], stride: Optional[Union[T, Tuple[T, ...]]] = None, padding: Union[T, Tuple[T, ...]] = 0, dilation: Union[T, Tuple[T, ...]] = 1, return_indices: bool = False, ceil_mode: bool = False)"

  class_name, options_map = parse(string4)

  dimensions = re.findall(r'\d+', class_name)
  dimensions = list(map(int, dimensions))
  print(options_map)

  create_baklava(class_name, options_map, dimensions)
  create_ir_node(class_name, options_map, dimensions)
