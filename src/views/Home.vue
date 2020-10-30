<template>
  <div class="home container-fluid d-flex flex-column">
    <Titlebar />
    <div class="row flex-grow-1">
      <div class="navbar-col">
        <Navbar />
      </div>
      <div class="col d-flex flex-column p-0">
        <Editor />
      </div>
    </div>
  </div>
</template>

<script lang="ts">
import { Component, Vue } from 'vue-property-decorator';
import Navbar from '@/components/navbar/Navbar.vue';
import Editor from '@/components/Editor.vue';
import Titlebar from '@/components/Titlebar.vue';
import { Mutation } from 'vuex-class';

@Component({
  components: {
    Titlebar,
    Navbar,
    Editor,
  },
})
export default class Home extends Vue {
  @Mutation('loadEditors') loadEditors!: (file: any) => void;

  created() {
    if (this.$cookies.isKey('unsaved')) {
      this.loadEditors(this.$cookies.get('unsaved'));
    }
  }
}
</script>

<style scoped>
.home {
  height: 100vh;
  overflow: auto;
}

.navbar-col {
  width: 3rem;
}
</style>
