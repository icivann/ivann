import re
import os

ir_folder = os.path.join("..", "src", "app", "ir", "pytorch model")
baklava_folder = os.path.join("..", "src", "nodes", "pytorch model")

type_map = {
  "int": "IntOption",
  "float": "SliderOption",
  "Union": "VectorOption",
  "Optional[Union]": "VectorOption",
  "Optional[T]": "VectorOption",
  "str": "DropdownOption",
  "bool": "TickBoxOption"
}

boolean_ = {
  "int": "bigint",
  "float": "number",
  "Union": "[T]",
  "Optional[T]": "[T]",
  "Optional[Union]": "[T]",
  "str": "enum",
  "bool": "boolean"
}
IR_map = boolean_

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
  f_path = os.path.join(baklava_folder, f"{class_name.capitalize()}Baklava.ts")
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
  type = Nodes.{class_name.capitalize()};
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
  f_path = os.path.join(ir_folder, f"{class_name.lower()}.ts")
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
      buildLine = f"get{field_name.capitalize()}(options.get({class_name}Options.{field_name.capitalize()}))"
    elif option_type == "bool":
      continue
    elif option_type == "[T]":

      buildLine = "["
      option_type = "["
      for i in range (dimensions[0]-1):
        buildLine+= f"  options.get({class_name}Options.{field_name.capitalize()}[{i}]), "
        option_type+= "bigint, "
      buildLine+= f"options.get({class_name}Options.{field_name.capitalize()})[{dimensions[0]-1}]], "
      option_type+= "bigint]"
    else:
      buildLine= f"options.get({class_name}Options.{field_name.capitalize()}),"


    fields.append(f"public readonly {field_name}: {option_type},")
    build.append(buildLine)
    pythonCode.append(f"{field_name}=${{this.{field_name}}}")

  fields = "\n  ".join(fields)
  build = "\n  ".join(build)
  pythonCode = ", ".join(pythonCode)

  contents = f"""import {{ {class_name}Options }} from '@/nodes/model/{class_name}Baklava';

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

  f_path = os.path.join("Nodes.txt")
  documentation = open(f_path, "r")
  documentation = documentation.read()
  nodes = documentation.split("\n")
  print(len(nodes))

  for i in range (len(nodes)):
    if nodes[i] == "":
      continue
    class_name, options_map = parse(nodes[i])
    print(i, class_name)

    dimensions = re.findall(r'\d+', class_name)
    dimensions = list(map(int, dimensions))

    print(dimensions)
    # create_baklava(class_name, options_map, dimensions)
    create_ir_node(class_name, options_map, dimensions)
