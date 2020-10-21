<template>
  <div class="left d-flex flex-column h-100 text-center">
    <!-- Build -->
    <div class="build tab-button" :class="isSelected(3)"
         @click="$store.commit('switchEditor', 3)">
      <i class="fas fa-hammer tab-icon"></i>
    </div>

    <div class="py-1 px-2">
      <hr/>
    </div>

    <!-- Model -->
    <div class="model tab-button" :class="isSelected(0)"
         @click="$store.commit('switchEditor', 0)">
      <img class="navbar-logo tab-icon" src="@/assets/images/nn_logo.png" alt="IVANN"/>
    </div>
    <div class="py-1 px-2">
      <hr/>
    </div>

    <!-- Data -->
    <div class="data tab-button" :class="isSelected(1)"
         @click="$store.commit('switchEditor', 1)">
      <i class="fas fa-database tab-icon"></i>
    </div>
    <div class="py-1 px-2">
      <hr/>
    </div>

    <!-- Train -->
    <div class="train tab-button" :class="isSelected(2)"
         @click="$store.commit('switchEditor', 2)">
      <i class="fas fa-cogs tab-icon"></i>
    </div>
  </div>
</template>

<script lang="ts">

import { Vue, Component } from 'vue-property-decorator';

@Component({})
export default class Navbar extends Vue {
  private isSelected(editor: number) {
    return (this.$store.state.editor === editor) ? 'selected' : '';
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
