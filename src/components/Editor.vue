<template>
  <div class="container-fluid d-flex flex-column">
    <div class="row flex-grow-1">
      <div class="col-9 px-0">
        <Canvas />
      </div>
      <div class="col-3 px-0">
        <Sidebar />
      </div>
    </div>
  </div>
</template>

<script>
import { Component, Vue } from 'vue-property-decorator';
import Sidebar from '@/components/Sidebar.vue';
import Canvas from '@/components/Canvas.vue';

@Component({
  components: {
    Sidebar,
    Canvas,
  },
})
export default class Editor extends Vue {
}
</script>

<style scoped>
</style>
