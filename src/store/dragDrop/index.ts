import { Module } from 'vuex';
import { DragState } from '@/store/dragDrop/types';
import { RootState } from '@/store/types';

export const dragDrop: Module<DragState, RootState> = {
  namespaced: false,
  state: {
    enableDrop: false,
  },
  getters: {
    /* Implementation can be extended, to prevent dropping on certain
     *  conditions. List of listeners/handlers? */
    canDrop: (state) => state.enableDrop,
  },
  actions: undefined,
  mutations: {
    enableDrop(state, value: boolean) {
      state.enableDrop = value;
    },
  },
};
