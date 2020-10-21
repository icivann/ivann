<template>
  <div class="container-fluid d-flex flex-column">
    <Resizable class="row">
      <div class="px-0 canvas-frame">
        <Canvas
          v-if="currEditorType === editorType.OVERVIEW"
          :viewPlugin="manager.overviewCanvas.viewPlugin"
          :key="editorType.OVERVIEW"
        />
        <Canvas
          v-else-if="currEditorType === editorType.MODEL"
          :viewPlugin="manager.modelCanvas.viewPlugin"
          :key="editorType.MODEL"
        />
        <Canvas
          v-else-if="currEditorType === editorType.DATA"
          :viewPlugin="manager.dataCanvas.viewPlugin"
          :key="editorType.DATA"
        />
        <Canvas
          v-else-if="currEditorType === editorType.TRAIN"
          :viewPlugin="manager.trainCanvas.viewPlugin"
          :key="editorType.TRAIN"
        />
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

  // created() {
  //   this.manager.modelCanvas;
  // }
}
</script>

<style scoped>
  .canvas-frame {
    width: 75%;
    max-width: 80%;
    min-width: 10%;
  }
</style>
