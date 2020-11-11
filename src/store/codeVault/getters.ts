import { GetterTree } from 'vuex';
import { RootState } from '@/store/types';
import { CodeVaultState } from '@/store/codeVault/types';

const codeVaultGetters: GetterTree<CodeVaultState, RootState> = {
  files: (state) => state.files,
  file: (state) => (filename: string) => state.files.find((file) => file.filename === filename),
  nodeTriggeringCodeVault: (state) => state.nodeTriggeringCodeVault,
};

export default codeVaultGetters;
