import { GetterTree } from 'vuex';
import { RootState } from '@/store/types';
import { CodeVaultState } from '@/store/codeVault/types';

const codeVaultGetters: GetterTree<CodeVaultState, RootState> = {
  files: (state) => state.files,
  filenames: (state) => {
    const names: Set<string> = new Set();
    state.files.forEach((file) => names.add(file.filename.slice(0, -3))); // remove '.py'
    return names;
  },
  file: (state) => (filename: string) => state.files.find((file) => file.filename === filename),
  nodeTriggeringCodeVault: (state) => state.nodeTriggeringCodeVault,
};

export default codeVaultGetters;
