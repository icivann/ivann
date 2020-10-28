import FunctionSignature from '@/app/parser/FunctionSignature';
import ParsedFunction from '@/app/parser/ParsedFunction';

/**
 * Given a Python file, returns the indentation character sequence
 * @param file file to get the indentation for
 */
function getIndentation(file: string): string | null {
  const lines = file.split('\n');

  for (let i = 0; i < lines.length; i += 1) {
    const line = lines[i];

    if (line.length > 0) {
      // Indent is with spaces
      if (line[0] === ' ') {
        return ' '.repeat(line.search(/\S/));
      }
      if (line[0] === '\t') { // Indent is with tabs
        return '\t'.repeat(line.search(/\S/));
      }
    }
  }

  return null;
}

function isLineFunctionDefinition(line: string): boolean {
  return line.startsWith('def ');
}

function isLineIndented(line: string, indentation: string): boolean {
  return line.startsWith(indentation);
}

function parseFunctionSignature(line: string): FunctionSignature | null {
  if (!line.startsWith('def ') || !line.includes('(') || !line.includes(')') || !line.includes(':')) {
    return null;
  }

  // remove def
  const name = line.substring(4, line.indexOf('(')).trim();
  const args = line.substring(line.indexOf('(') + 1, line.indexOf(')'));

  return new FunctionSignature(
    name,
    args.length > 0 ? args.split(',').map((it) => it.trim()) : [],
  );
}

function parse(str: string): Promise<ParsedFunction[]> {
  return new Promise((resolve, reject) => {
    const indentation = getIndentation(str);

    // If we failed to get the indentation level
    if (indentation === null) {
      reject(new Error('badly indented file'));
      return;
    }

    // Split lines and add an empty line at the end
    const lines = str.split('\n');
    lines.push('');

    let isParsingFunction = false;
    let currentSignature: FunctionSignature | null = null;
    let currentBody: string | null = null;

    const parsedFunctions = new Array<ParsedFunction>();

    for (let i = 0; i < lines.length; i += 1) {
      const line = lines[i];

      if (isParsingFunction) {
        if (isLineIndented(line, indentation)) {
          if (!isParsingFunction) {
            reject(new Error(`unexpected indentation at line ${i}: ${line}`));
          } else {
            // Parsing function body
            currentBody += `${line}\n`;
          }
        } else {
          // Finished parsing function
          parsedFunctions.push(new ParsedFunction(
            currentSignature!.name,
            currentBody!,
            currentSignature!.args,
          ));

          isParsingFunction = false;
          currentSignature = null;
          currentSignature = null;
        }
      } else {
        if (isLineFunctionDefinition(line)) {
          const signature = parseFunctionSignature(line);

          if (signature === null) {
            reject(new Error(`failed to parse function definition at line ${i}: ${line}`));
            return;
          }

          isParsingFunction = true;
          currentSignature = signature;
          currentBody = '';
        }

        if (isLineIndented(line, indentation)) {
          reject(new Error(`unexpected indentation at line ${i}: ${line}`));
        }
      }
    }

    resolve(parsedFunctions);
  });
}

export default parse;
