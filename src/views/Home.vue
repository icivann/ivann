<template>
  <div class="home container-fluid d-flex flex-column">
    <div class="row">
      <h1>Sidebar</h1>
    </div>
    <div class="row flex-grow-1">
      <div class="col-1 px-0">
        <LeftSidebar/>
      </div>
      <div class="col-9 px-0">
        <NodeEditor/>
      </div>
      <div class="col-2 px-0">
        <RightSidebar/>
      </div>
    </div>
  </div>
</template>

<script>
import { Component, Vue } from 'vue-property-decorator';
import LeftSidebar from '@/components/LeftSidebar.vue';
import NodeEditor from '@/components/NodeEditor.vue';
import RightSidebar from '@/components/RightSidebar.vue';

@Component({
  components: {
    LeftSidebar,
    NodeEditor,
    RightSidebar,
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
</style>
