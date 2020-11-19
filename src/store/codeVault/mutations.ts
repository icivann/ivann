import { MutationTree } from 'vuex';
import { CodeVaultState, ParsedFile } from '@/store/codeVault/types';
import ParsedFunction from '@/app/parser/ParsedFunction';
import Custom from '@/nodes/common/Custom';

const codeVaultMutations: MutationTree<CodeVaultState> = {
  resetState(state) {
    state.files = [];
  },
  loadFiles(state, files: ParsedFile[]) {
    const newFiles: ParsedFile[] = [];
    for (const file of files) {
      newFiles.push({
        filename: file.filename,
        functions: file.functions.map(
          /* JSON parses functions to an interface, not to the ParsedFunction object. */
          (func) => new ParsedFunction(func.name, func.body, func.args),
        ),
        open: false,
      });
    }
    state.files = newFiles;
  },
  addFile(state, file: ParsedFile) {
    state.files.push(file);
  },
  deleteFile(state, filename: string) {
    state.files = state.files.filter((file) => file.filename !== filename);
  },
  setFile(state, file: ParsedFile) {
    for (const stateFile of state.files) {
      if (stateFile.filename === file.filename) stateFile.functions = file.functions;
    }
  },
  addFunc(state, { filename, func }: { filename: string; func: ParsedFunction }) {
    state.files = state.files.map((file) => {
      if (file.filename === filename) {
        file.functions.push(func);
      }
      return file;
    });
  },
  deleteFunc(state, { filename, func }: { filename: string; func: ParsedFunction }) {
    state.files = state.files.map((file) => {
      if (file.filename === filename) {
        return { ...file, functions: file.functions.filter((fileFunc) => fileFunc !== func) };
      }
      return file;
    });
  },
  closeFiles(state) {
    state.files = state.files.map((file) => ({ ...file, open: false }));
  },
  closeFile(state, filename: string) {
    for (const file of state.files) {
      if (file.filename === filename) file.open = false;
    }
  },
};

export default codeVaultMutations;
