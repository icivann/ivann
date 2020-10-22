<template>
  <div class="container-fluid d-flex flex-column">
    <Resizable class="row">
      <div class="px-0 canvas-frame">
        <Canvas :viewPlugin="this.viewPlugin()"/>
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
    'overviewEditor',
    'modelEditor',
    'dataEditor',
    'trainEditor',
  ]),
})
export default class Editor extends Vue {
  private editorType = EditorType;

  private manager: EditorManager = EditorManager.getInstance();

  private viewPlugin(): ViewPlugin | undefined {
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
    width: 75%;
    max-width: 80%;
    min-width: 10%;
  }
</style>
