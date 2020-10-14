<template>
  <div class="Navbar row py-2">
    <div class="col text-left ml-2">
      <img class="navbar-logo mr-2" src="@/assets/images/nn_logo.png" alt="IVANN" />
      <span class="text">IVANN</span>
    </div>
    <div class="col text-center">
      <span class="text">
        MNIST-Demo
      </span>
    </div>
    <div class="col text-right">
      <i class="navbar-icon fas fa-share-alt fa-lg"></i>
      <i class="navbar-icon fas fa-folder-open fa-lg"></i>
      <i class="navbar-icon fas fa-save fa-lg"></i>
    </div>
  </div>
</template>

<script lang="ts">
import { Component, Vue } from 'vue-property-decorator';

@Component
export default class Navbar extends Vue {
}
</script>

<style lang="scss" scoped>
.Navbar {
  height: 2.5rem;
  background-color: #303030;

  border-bottom: 1px solid #ececec;
}

.navbar-logo {
  height: 1.8rem;
}

.text {
  color: #ececec;
}

.navbar-icon {
  margin-left: 0.25rem;
  margin-right: 0.25rem;
  color: #ececec;

  &:hover {
    color: darkblue;
  }
}
</style>
