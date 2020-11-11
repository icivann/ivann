import { MutationTree } from 'vuex';
import { CodeVaultState, ParsedFile } from '@/store/codeVault/types';
import ParsedFunction from '@/app/parser/ParsedFunction';
import Custom from '@/nodes/model/custom/Custom';

const codeVaultMutations: MutationTree<CodeVaultState> = {
  resetState(state) {
    state.files = [];
  },
  loadFiles(state, files: ParsedFile[]) {
    state.files = files;
  },
  addFile(state, file: ParsedFile) {
    state.files.push(file);
  },
  deleteFile(state, filename: string) {
    state.files = state.files.filter((file) => file.filename !== filename);
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
  linkNode(state, node?: Custom) {
    state.nodeTriggeringCodeVault = node;
  },
};

export default codeVaultMutations;
