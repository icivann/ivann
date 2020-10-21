<template>
  <div class="left d-flex flex-column h-100 text-center">
    <!-- Build -->
    <div class="build tab-button" :class="isSelected(editorType.OVERVIEW)"
         @click="switchEditor({editorType: editorType.OVERVIEW, index: 0})"> <!-- TODO onClick -->
      <i class="fas fa-hammer tab-icon"></i>
    </div>

    <div class="py-1 px-2">
      <hr/>
    </div>

    <!-- Model -->
    <div class="model tab-button" :class="isSelected(editorType.MODEL)"
         @click="switchEditor({editorType: editorType.MODEL, index: 0})"> <!-- TODO onClick -->
      <img class="navbar-logo tab-icon" src="@/assets/images/nn_logo.png" alt="IVANN"/>
    </div>
    <div class="py-1 px-2">
      <hr/>
    </div>

    <!-- Data -->
    <div class="data tab-button" :class="isSelected(editorType.DATA)"
         @click="switchEditor({editorType: editorType.DATA, index: 0})"> <!-- TODO onClick -->
      <i class="fas fa-database tab-icon"></i>
    </div>
    <div class="py-1 px-2">
      <hr/>
    </div>

    <!-- Train -->
    <div class="train tab-button" :class="isSelected(editorType.TRAIN)"
         @click="switchEditor({editorType: editorType.TRAIN, index: 0})"> <!-- TODO onClick -->
      <i class="fas fa-cogs tab-icon"></i>
    </div>
  </div>
</template>

<script lang="ts">

import { Vue, Component } from 'vue-property-decorator';
import EditorType from '@/EditorType';
import { mapActions, mapMutations } from 'vuex';

@Component({
  methods: mapMutations([
    'switchEditor',
    'newEditor',
  ]),
})
export default class Navbar extends Vue {
  private editorType = EditorType;

  private isSelected(editorType: EditorType) {
    return (this.$store.state.currEditorType === editorType) ? 'selected' : '';
  }
}

</script>

<style scoped>
  .left {
    background: var(--background);
    color: var(--foreground);
    border-right: 0.08rem solid var(--grey);
  }

  .tab-button {
    padding: 0.75rem 0;
    transition-duration: 0.1s;
    border-left-style: solid;
    border-left-width: 1px;
    border-left-color: var(--background);
  }

  .tab-button:hover {
    background: #1c1c1c;
    transition-duration: 0.1s;
    border-left-color: var(--blue);
    cursor: pointer;
  }

  .tab-button.selected {
    border-left-width: 4px;
    border-left-color: var(--blue);
  }

  .tab-icon {
    font-size: 1.5rem;
    height: 1.5rem;
  }

  hr {
    border-top: 0.1rem solid var(--dark-grey) !important;
  }

  .build {
    margin-top: 1rem;
  }
</style>
