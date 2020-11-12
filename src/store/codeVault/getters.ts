import { GetterTree } from 'vuex';
import { RootState } from '@/store/types';
import { CodeVaultState, FilenamesList } from '@/store/codeVault/types';

const codeVaultGetters: GetterTree<CodeVaultState, RootState> = {
  files: (state) => state.files,
  filenames: (state) => {
    const names: Set<string> = new Set();
    state.files.forEach((file) => names.add(file.filename));
    return names;
  },
  file: (state) => (filename: string) => state.files.find((file) => file.filename === filename),
  nodeTriggeringCodeVault: (state) => state.nodeTriggeringCodeVault,
  fileIndexFromFilename: (state) => (filename: string) => state
    .files.findIndex((file) => file.filename === filename),
  functionIndexFromFunctionName: (state) => (fileIndex: number, functionName: string) => state
    .files[fileIndex].functions.findIndex((func) => func.name === functionName),
  filenamesList: (state): FilenamesList => ({
    filenames: state.files.map((file) => file.filename),
  }),
};

export default codeVaultGetters;
