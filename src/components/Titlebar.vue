<template>
  <div class="titlebar row py-2">
    <div class="col text-left">
      <img class="img-fluid titlebar-logo mr-2" src="@/assets/images/nn_logo.png" alt="IVANN"/>
      <span class="text">IVANN</span>
    </div>
    <div class="col text-center">
      <span class="text">
        MNIST-Demo
      </span>
    </div>
    <div class="col text-right">
      <span class="icon-button" @click="share">
        <i class="titlebar-icon fas fa-share-alt fa-lg mx-2"/>
      </span>
      <span class="icon-button" @click="open">
        <i class="titlebar-icon fas fa-folder-open fa-lg mx-2"/>
      </span>
      <span class="icon-button" @click="save">
        <i class="titlebar-icon fas fa-save fa-lg mx-2"/>
      </span>
    </div>
  </div>
</template>

<script lang="ts">
import { Component, Vue } from 'vue-property-decorator';

@Component
export default class Titlebar extends Vue {
  share() {
    console.log(`Share button pressed. ${this.$data}`);
  }

  open() {
    console.log(`Open button pressed. ${this.$data}`);
  }

  save() {
    console.log(`Save button pressed. ${this.$data}`);
  }
}
</script>

<style lang="scss" scoped>
  .titlebar {
    height: 2.5rem;
    background-color: var(--background-alt);

    border-bottom: 0.08rem solid var(--grey);
  }

  .titlebar-logo {
    height: 1.2rem;
  }

  .text {
    color: var(--foreground);
  }

  .titlebar-icon {
    color: var(--foreground);
  }

  .icon-button {
    background-color: var(--background-alt);
    width: 5rem;
    height: 5rem;
    margin-left: 0.15rem;
    margin-right: 0.15rem;
    padding: 0.3rem 0.1rem;
    position: relative;
    top: 0;

    &:hover {
      background-color: #2c2c2c;
      cursor: pointer;
    }
  }
</style>
