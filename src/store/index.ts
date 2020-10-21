import Vue from 'vue';
import Vuex from 'vuex';

Vue.use(Vuex);

export default new Vuex.Store({
  state: {
    editor: 0,
  },
  mutations: {
    switchEditor(state, newEditor) {
      state.editor = newEditor;
    },
  },
  actions: {
  },
  modules: {
  },
});
