<template>
  <div class="container-fluid d-flex flex-column">
    <Resizable class="row">
      <div class="px-0 canvas-frame">
        <Canvas v-if="$store.state.editor === 0"
                :editor="manager.modelBaklavaEditor"/>
        <Canvas v-else-if="$store.state.editor === 1"
                :editor="manager.dataBaklavaEditor"/>
        <Canvas v-else-if="$store.state.editor === 2"
                :editor="manager.trainBaklavaEditor"/>
        <Canvas v-else-if="$store.state.editor === 3"
                :editor="manager.overviewBaklavaEditor"/>
      </div>
      <Resizer/>
      <div class="px-0 flex-grow-1">
        <Sidebar/>
      </div>
    </Resizable>
  </div>
</template>

<script lang="ts">
import { Component, Vue } from 'vue-property-decorator';
import Sidebar from '@/components/Sidebar.vue';
import Canvas from '@/components/Canvas.vue';
import EditorManager from '@/EditorManager';
import Resizer from '@/components/Resize/Resizer.vue';
import Resizable from '@/components/Resize/Resizable.vue';

@Component({
  components: {
    Resizer,
    Resizable,
    Sidebar,
    Canvas,
  },
})
export default class Editor extends Vue {
  private manager: EditorManager = EditorManager.getInstance();
}
</script>

<style scoped>
  .canvas-frame {
    width: 75%;
    max-width: 80%;
    min-width: 10%;
  }
</style>
