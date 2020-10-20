<template>
  <div class="home container-fluid d-flex flex-column">
    <Navbar />
    <div class="row flex-grow-1">
      <div class="left-sidebar-col">
        <LeftSidebar />
      </div>
      <div class="col d-flex flex-column p-0">
        <Editor />
      </div>
    </div>
  </div>
</template>

<script>
import { Component, Vue } from 'vue-property-decorator';
import LeftSidebar from '@/components/LeftSidebar.vue';
import Editor from '@/components/Editor.vue';
import Navbar from '@/components/Navbar.vue';

@Component({
  components: {
    Navbar,
    LeftSidebar,
    Editor,
  },
})
export default class Home extends Vue {
}
</script>

<style scoped>
.home {
  height: 100vh;
  overflow: auto;
}

.left-sidebar-col {
  width: 3rem;
}
</style>
