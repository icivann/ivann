import { Module } from 'vuex';
import { RootState } from '@/store/types';
import { CodeVaultState } from '@/store/codeVault/types';
import codeVaultGetters from '@/store/codeVault/getters';
import codeVaultMutations from '@/store/codeVault/mutations';

export const codeVaultState: CodeVaultState = {
  files: [],
};

export const codeVault: Module<CodeVaultState, RootState> = {
  namespaced: false,
  state: codeVaultState,
  getters: codeVaultGetters,
  actions: undefined,
  mutations: codeVaultMutations,
};
