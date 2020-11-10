import ParsedFunction from '@/app/parser/ParsedFunction';

export const CustomFilename = 'Notepad';

export interface ParsedFile {
  filename: string;
  functions: ParsedFunction[];
}

export interface CodeVaultState {
  files: ParsedFile[];
}
