import { Module } from 'vuex';
import { RootState } from '@/store/types';
import { CodeVaultState, CustomFilename } from '@/store/codeVault/types';
import codeVaultGetters from '@/store/codeVault/getters';
import codeVaultMutations from '@/store/codeVault/mutations';

export const codeVaultState: CodeVaultState = {
  files: [{
    filename: CustomFilename,
    functions: [],
  }],
  nodeTriggeringCodeVault: undefined,
};

export const codeVault: Module<CodeVaultState, RootState> = {
  namespaced: false,
  state: codeVaultState,
  getters: codeVaultGetters,
  actions: undefined,
  mutations: codeVaultMutations,
};
