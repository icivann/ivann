import ParsedFunction from '@/app/parser/ParsedFunction';
import Custom from '@/nodes/common/Custom';

export interface ParsedFile {
  filename: string;
  functions: ParsedFunction[];
  open: boolean;
}

export interface CodeVaultState {
  files: ParsedFile[];
}

export interface FilenamesList {
  filenames: string[];
}
