import re
import os

ir_folder = os.path.join("..", "src", "app", "ir", "model")
baklava_folder = os.path.join("..", "src", "nodes", "model")

type_map = {
  "int": "IntOption",
  "float": "SliderOption",
  "Union": "VectorOption",
  "Optional[Union]": "VectorOption",
  "Optional[T]": "VectorOption",
  "Optional[Any]": "VectorOption",
  "str": "DropdownOption",
  "bool": "TickBoxOption"
}

boolean_ = {
  "int": "bigint",
  "float": "number",
  "Union": "[T]",
  "Optional[T]": "[T]",
  "Optional[Union]": "[T]",
  "Optional[Any]": "[T]",
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


    ret[name] = (x_type, default_value)
    name = x_split[0]

  if not split:
    return ret

  x_split = split[-1]
  x_split = x_split.replace(" ", "")
  rest = x_split.split("=")
  default_value = rest[1] if len(rest) > 1 else None
  x_type = rest[0]


  ret[name] = (x_type, default_value)

  return ret


def parse(x):
  x = x[9:]

  split = x.split('(', 1)
  class_name = split[0]
  parameter_name_and_types = split[1][:-1]

  parameter_map = parse_options(parameter_name_and_types)

  return class_name, parameter_map


def create_baklava(class_name, option_map: dict, dim):
  f_path = os.path.join(baklava_folder, f"{class_name.capitalize()}.ts")
  baklava_file = open(f_path, "w")

  option_enums_header = f"export enum {class_name}Options {{"
  option_enums_values = []
  options_add = []
  for k, v in option_map.items():
    option_name = k.capitalize()
    option_name = ''.join(map(lambda s: s.strip().capitalize(), option_name.split('_')))
    enumValue = k.replace("_", " ").capitalize()
    option_enums_values.append(f"{option_name} = '{enumValue}'")

    option_type, default_value = v
    option_type = re.sub('\[(.*?)\]', '[T]', option_type)
    option_type = type_map[option_type]

    if default_value == 'None':
      # todo: check here: none
      default_value = '0'

    default_value = default_value if default_value else 0

    if default_value == 'True':
      default_value = 'CheckboxValue.CHECKED'
    elif default_value == 'False':
      default_value = 'CheckboxValue.UNCHECKED'

    elif len(dim)>0:
      default_value_vector = '['
      for i in range(dimensions[0] - 1):
        default_value_vector += f'{default_value}, '
      default_value_vector += f'{default_value}'
      default_value_vector += ']'
      default_value = default_value_vector

    option_add = f"""this.addOption({class_name}Options.{option_name}, TypeOptions.{option_type}, {default_value});"""
    options_add.append(option_add)

  options_add = "\n  ".join(options_add)
  option_enums_values = ",\n  ".join(option_enums_values)
  option_enums = f"{option_enums_header}\n  {option_enums_values}\n}}"
  # option_enums = "\n".join(option_enums)


  contents = f"""import {{ Node }} from '@baklavajs/core';
import {{ ModelNodes }} from '@/nodes/model/Types';
import {{ TypeOptions }} from '@/nodes/model/BaklavaDisplayTypeOptions';
import CheckboxValue from '@/baklava/CheckboxValue';

{option_enums}
export default class {class_name} extends Node {{
  type = ModelNodes.{class_name};
  name = ModelNodes.{class_name};

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
  pythonCode = []
  enum_import = 'nodeName'

  fields.append("public readonly name: string,")
  build.append("\n  options.get(nodeName),")
  for k, v in option_map.items():
    field_name = k
    field_name = ''.join(map(lambda s: s.strip().capitalize(), field_name.split('_')))
    option_type, default_value = v
    option_type = re.sub('\[(.*?)\]', '[T]', option_type)
    option_type = IR_map[option_type]

    if option_type == "enum":
      enum_import += f', {field_name}, get{field_name}'
      option_type = field_name
      buildLine = f"get{field_name.capitalize()}(options.get({class_name}Options.{field_name})),"
    elif option_type == "bool":
      continue
    elif option_type == "[T]":
      dim = dimensions[0] - 1 if len(dimensions) > 0 else 0
      buildLine = "["
      option_type = "["
      for i in range(dim):
        buildLine += f"  options.get({class_name}Options.{field_name})[{i}], "
        option_type += "bigint, "
      buildLine += f"options.get({class_name}Options.{field_name})[{dim}]], "
      option_type += "bigint]"
    else:
      buildLine = f"options.get({class_name}Options.{field_name}),"

    fields.append(f"public readonly {field_name}: {option_type},")
    build.append(buildLine)
    pythonCode.append(f"{field_name}=")

    if len(dimensions) > 0 and dimensions[0] > 1:
      pythonCode.append(f"(${{this.{field_name}}})")
    elif option_type == "str":
      pythonCode.append(f"'${{this.{field_name}}}'")
    else:
      pythonCode.append(f"${{this.{field_name}}}")

  fields = "\n  ".join(fields)
  build = "\n  ".join(build)
  pythonCode = ", ".join(pythonCode)

  contents = f"""import {{ {class_name}Options }} from '@/nodes/model/{class_name.capitalize()}';
import {{ {enum_import} }} from '@/app/ir/irCommon';

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


def create_node_reg(class_name):

  content = f"""
      {{ 
        name: ModelNodes.{class_name},
        node: {class_name},
      }},
  """
  with open("node_reg.txt", "a") as myfile:
    myfile.write(content)

def create_mode_node_type(class_name):

  content = f"""
        {class_name} = '{class_name}',"""
  with open("node_modelNode_type.txt", "a") as myfile:
    myfile.write(content)


def create_mapping(class_name):

  content = f"""
    ['{class_name}', {class_name}.build],"""
  with open("node_mapping.txt", "a") as myfile:
    myfile.write(content)

if __name__ == "__main__":

  f_path = os.path.join("Nodes.txt")
  documentation = open(f_path, "r")
  documentation = documentation.read()
  nodes = documentation.split("\n")


  for i in range(len(nodes)):
    if nodes[i] == "":
      continue
    class_name, options_map = parse(nodes[i])

    dimensions = re.findall(r'\d+', class_name)
    dimensions = list(map(int, dimensions))

    create_baklava(class_name, options_map, dimensions)
    create_ir_node(class_name, options_map, dimensions)

    create_node_reg(class_name)
    create_mode_node_type(class_name)
    create_mapping(class_name)


    print(class_name)
