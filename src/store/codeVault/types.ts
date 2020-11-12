import ParsedFunction from '@/app/parser/ParsedFunction';
import Custom from '@/nodes/model/custom/Custom';

export interface ParsedFile {
  filename: string;
  functions: ParsedFunction[];
}

export interface CodeVaultState {
  files: ParsedFile[];
  nodeTriggeringCodeVault?: Custom;
}