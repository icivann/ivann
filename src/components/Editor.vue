<template>
  <div class="container-fluid d-flex flex-column">
    <Resizable class="row" @width-change="changeWidth">
      <div class="px-0 canvas-frame"
           :style="`width: min(calc(100vw - 3rem - ${sidebarWidth}px), ${editorWidth}%)`">
        <Canvas
          :viewPlugin="this.currViewPlugin()"
          :editorModel="currEditorModel"
        />
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
import { mapGetters } from 'vuex';
import EditorType from '@/EditorType';
import { ViewPlugin } from '@baklavajs/plugin-renderer-vue';

@Component({
  components: {
    Resizer,
    Resizable,
    Sidebar,
    Canvas,
  },
  computed: mapGetters([
    'currEditorType',
    'currEditorModel',
    'overviewEditor',
    'modelEditor',
    'dataEditor',
    'trainEditor',
  ]),
})
export default class Editor extends Vue {
  private manager: EditorManager = EditorManager.getInstance();
  private editorWidth = 75; /* percentage */
  private sidebarWidth = 280; /* pixels */

  private changeWidth(percentage: number) {
    this.editorWidth = percentage;
  }

  private currViewPlugin(): ViewPlugin | undefined {
    switch (this.$store.getters.currEditorType) {
      case EditorType.OVERVIEW:
        return this.manager.overviewCanvas.viewPlugin;
      case EditorType.MODEL:
        return this.manager.modelCanvas.viewPlugin;
      case EditorType.DATA:
        return this.manager.dataCanvas.viewPlugin;
      case EditorType.TRAIN:
        return this.manager.trainCanvas.viewPlugin;
      default:
        return undefined;
    }
  }
}
</script>

<style scoped>
  .canvas-frame {
    min-width: 20%;
  }
</style>
