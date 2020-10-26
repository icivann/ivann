<template>
  <div class="container-fluid d-flex flex-column">
    <Resizable class="row" @width-change="changeWidth">
      <div class="px-0 canvas-frame"
           :style="`width: min(calc(100vw - 3rem - ${sidebarWidth}px), ${editorWidth}%)`">
        <Canvas
          :viewPlugin="this.manager.viewPlugin"
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
import { Getter } from 'vuex-class';

@Component({
  components: {
    Resizer,
    Resizable,
    Sidebar,
    Canvas,
  },
  computed: mapGetters(['currEditorModel']),
})
export default class Editor extends Vue {
  private manager: EditorManager = EditorManager.getInstance();
  private editorWidth = 75; /* percentage */
  private sidebarWidth = 280; /* pixels */
  @Getter('currEditorType') currEditorType!: EditorType;

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
