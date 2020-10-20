<template>
  <div class="container-fluid d-flex flex-column">
    <div class="row flex-grow-1">
      <div class="col-9 px-0">
        <Canvas v-if="$store.state.editor === model"
                :editor="manager.modelBaklavaEditor"/>
        <Canvas v-else-if="$store.state.editor === data"
                :editor="manager.dataBaklavaEditor"/>
        <Canvas v-else-if="$store.state.editor === training"
                :editor="manager.trainBaklavaEditor"/>
        <Canvas v-else-if="$store.state.editor === overview"
                :editor="manager.overviewBaklavaEditor"/>
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
  data: {
    model: EditorType.MODEL,
    data: EditorType.DATA,
    training: EditorType.TRAIN,
    overview: EditorType.OVERVIEW,
  },
})
export default class Editor extends Vue {
  private manager: EditorManager = EditorManager.getInstance();
}
</script>

<style scoped>
</style>
