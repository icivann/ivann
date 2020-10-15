<template>
  <div class="home container-fluid d-flex flex-column">
    <Navbar />
    <div class="row flex-grow-1">
      <div class="left-sidebar-col">
        <LeftSidebar />
      </div>
      <div class="col d-flex flex-column">
        <div class="row flex-grow-1">
          <div class="col-9 px-0">
            <NodeEditor />
          </div>
          <div class="col-3 px-0">
            <RightSidebar />
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { Component, Vue } from 'vue-property-decorator';
import LeftSidebar from '@/components/LeftSidebar.vue';
import NodeEditor from '@/components/NodeEditor.vue';
import Navbar from '@/components/Navbar.vue';
import RightSidebar from '@/components/RightSidebar.vue';

@Component({
  components: {
    Navbar,
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

.left-sidebar-col {
  width: 3rem;
}
</style>
