<template>
  <div class="titlebar row py-2">
    <div class="col text-left">
      <img class="img-fluid titlebar-logo mr-2" src="@/assets/images/nn_logo.png" alt="IVANN" />
      <span class="text">IVANN</span>
    </div>
    <div class="col text-center">
      <span class="text">
        MNIST-Demo
      </span>
    </div>
    <div class="col text-right">
      <i class="titlebar-icon fas fa-share-alt fa-lg mx-2"></i>
      <i class="titlebar-icon fas fa-folder-open fa-lg mx-2"></i>
      <i class="titlebar-icon fas fa-save fa-lg mx-2"></i>
    </div>
  </div>
</template>

<script lang="ts">
import { Component, Vue } from 'vue-property-decorator';

@Component
export default class Titlebar extends Vue {
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
  margin-left: 0.25rem;
  margin-right: 0.25rem;
  color: var(--foreground);

  &:hover {
    color: var(--blue);
  }
}
</style>
