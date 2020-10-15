<template>
  <div class="Navbar row py-2">
    <div class="col text-left">
      <img class="navbar-logo mr-2" src="@/assets/images/nn_logo.png" alt="IVANN" />
      <span class="text">IVANN</span>
    </div>
    <div class="col text-center">
      <span class="text">
        MNIST-Demo
      </span>
    </div>
    <button @click="save"> Export Model </button>
    <div class="col text-right">
      <i class="navbar-icon fas fa-share-alt fa-lg"></i>
      <i class="navbar-icon fas fa-folder-open fa-lg"></i>
      <i class="navbar-icon fas fa-save fa-lg" ></i>
    </div>
  </div>
</template>

<script lang="ts">
import { Component, Vue } from 'vue-property-decorator';
import { mnist } from '@/app/ir/Graphs';
import { generateKeras } from '@/app/generators/keras/kerasGenerator';
import { FileSaver } from 'file-saver-typescript';

@Component
export default class Navbar extends Vue {
  save_msg = 'Saving...';

  private save() {
    const temp = this.save_msg; // TODO: to stop eslint from crying

    const model = mnist();
    const code = generateKeras(model);
    console.log(code);

    const fileSaver: any = new FileSaver();
    fileSaver.responseData = code;
    fileSaver.strFileName = 'model.py';
    fileSaver.strMimeType = 'text/plain';
    fileSaver.initSaveFile();
    console.log('saved..');
  }
}
</script>

<style lang="scss" scoped>
.Navbar {
  height: 2.5rem;
  background-color: var(--background-alt);

  border-bottom: 1px solid var(--foreground);
}

.navbar-logo {
  height: 1.2rem;
}

.text {
  color: var(--foreground);
}

.navbar-icon {
  margin-left: 0.25rem;
  margin-right: 0.25rem;
  color: var(--foreground);

  &:hover {
    color: var(--blue);
  }
}
</style>
