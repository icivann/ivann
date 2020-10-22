<template>
  <div class="container-fluid d-flex flex-column">
    <Resizable class="row" @width-change="changeWidth">
      <div class="px-0 canvas-frame"
           :style="`width: min(calc(100vw - 3rem - ${sidebarWidth}px), ${editorWidth}%)`">
        <Canvas v-show="$store.state.editor === 0"
                :editor="manager.modelBaklavaEditor"
                :abstract-canvas="manager.modelCanvas"/>
        <Canvas v-show="$store.state.editor === 1"
                :editor="manager.dataBaklavaEditor"
                :abstract-canvas="manager.dataCanvas"/>
        <Canvas v-show="$store.state.editor === 2"
                :editor="manager.trainBaklavaEditor"
                :abstract-canvas="manager.trainCanvas"/>
        <Canvas v-show="$store.state.editor === 3"
                :editor="manager.overviewBaklavaEditor"
                :abstract-canvas="manager.overviewCanvas"/>
      </div>
      <Resizer/>
      <div class="px-0 flex-grow-1"
           :style="`max-width: max(${sidebarWidth}px, calc(100% - ${editorWidth}%))`">
        <Sidebar/>
      </div>
    </Resizable>
  </div>
</template>

<script lang="ts">
import { Component, Vue } from 'vue-property-decorator';
import Sidebar from '@/components/Sidebar.vue';
import Canvas from '@/components/canvas/Canvas.vue';
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
  private editorWidth = 75; /* percentage */
  private sidebarWidth = 280; /* pixels */

  private changeWidth(percentage: number) {
    this.editorWidth = percentage;
  }
}
</script>

<style scoped>
  .canvas-frame {
    min-width: 20%;
  }
</style>
