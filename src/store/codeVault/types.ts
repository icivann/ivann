import ParsedFunction from '@/app/parser/ParsedFunction';
import Custom from '@/nodes/common/Custom';

export interface ParsedFile {
  filename: string;
  functions: ParsedFunction[];
}

export interface CodeVaultState {
  files: ParsedFile[];
  nodeTriggeringCodeVault?: Custom;
}

export interface FilenamesList {
  filenames: string[];
}
