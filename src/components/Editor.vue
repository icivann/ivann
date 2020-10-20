<template>
  <div class="container-fluid d-flex flex-column">
    <div class="row flex-grow-1">
      <div class="col-9 px-0">
        <Canvas v-if="$store.state.editor === 0"
                :editor="manager.modelBaklavaEditor"/>
        <Canvas v-else-if="$store.state.editor === 1" :editor="manager.dataBaklavaEditor"/>
        <Canvas v-else-if="$store.state.editor === 2" :editor="manager.trainBaklavaEditor"/>
        <Canvas v-else-if="$store.state.editor === 3" :editor="manager.overviewBaklavaEditor"/>
      </div>
      <div class="col-3 px-0">
        <Sidebar/>
      </div>
    </div>
  </div>
</template>

<script lang="ts">
import { Component, Vue } from 'vue-property-decorator';
import Sidebar from '@/components/Sidebar.vue';
import Canvas from '@/components/Canvas.vue';
import EditorManager from '@/EditorManager';
import EditorType from '@/EditorType';

@Component({
  components: {
    Sidebar,
    Canvas,
  },
})
export default class Editor extends Vue {
  private manager: EditorManager = EditorManager.getInstance();
}
</script>

<style scoped>
</style>
