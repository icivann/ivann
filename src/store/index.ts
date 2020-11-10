import Vue from 'vue';
import Vuex, { StoreOptions } from 'vuex';
import { RootState } from '@/store/types';
import { editors } from '@/store/editors';
import { codeVault } from '@/store/codeVault';

Vue.use(Vuex);

const store: StoreOptions<RootState> = {
  state: {
    showTutorial: false,
  },
  getters: {
  },
  actions: {
  },
  mutations: {
  },
  modules: {
    editors,
    codeVault,
  },
};

export default new Vuex.Store<RootState>(store);
