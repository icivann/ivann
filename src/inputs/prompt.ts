// Returns unique (non-empty) name entered or null if cancelled
export function uniqueTextInput(inputs: Set<string>, msg: string) {
  let name: string | null = null;
  let isNameUnique = false;
  while (!isNameUnique) {
    name = prompt(msg);

    // Name is null if cancelled
    if (name === null) break;

    // Loop until unique non-empty name has been entered
    if (name !== '' && !inputs.has(name)) isNameUnique = true;
  }

  return name;
}
